import argparse
import copy
from dataclasses import dataclass
import gc
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import plotly.express as px
import torch
from torchaudio.datasets import LIBRISPEECH
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from StreamASRLib.model import RawMelAdapter
from StreamASRLib.training import read_config
from StreamASRLib.dataset import MergeDataset, LibriDataset, LibriLogMel
from StreamASRLib.eval import calculate_wer

from torchtune.modules import RotaryPositionalEmbeddings

import pandas as pd

WANDB_TOKEN = os.environ['WANDB_TOKEN']

QWEN_REPO = 'Qwen/Qwen2.5-0.5B'
QWEN_EMB_SIZE = 896

EOS_TOKEN = 151643
TEXT_PADDING_TOKEN = EOS_TOKEN

EXT_TOKEN_BASE = 151936
BOA_TOKEN = EXT_TOKEN_BASE + 1
EOA_TOKEN = EXT_TOKEN_BASE + 2

# Set torch seed
torch.manual_seed(239)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_qwen_tokenizer(use_fast: bool = True) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(QWEN_REPO, use_fast=use_fast)


def calculate_grad_norm(model):
    return torch.sqrt(sum([torch.norm(param.grad) ** 2 for param in model.parameters()]))


class Reporter:
    def __init__(self, config, wandb_enabled: bool = True):
        self.config = config
        self.wandb_enabled = wandb_enabled

    def setup(self):
        if self.config.wandb.enabled:
            print('Initialising W&B')
            wandb.login(key=WANDB_TOKEN)
            wandb.init(
                project=self.config.wandb.project,
                config=self.config,
                name=self.config.wandb.run_name,
                notes=self.config.wandb.notes,
                # disable system logging
                settings=wandb.Settings(
                    _disable_stats=True, _disable_meta=True)
            )
        else:
            print('Warning: wandb is not enabled')

    def report_train_loss(self, step: int, loss: float, info: Optional[Dict[str, Any]] = None):
        if info is None:
            info = {}
        print(f'Step {step}; train loss: {loss}; info: {info}')

        if not self.wandb_enabled:
            return
        log = {'train_loss': loss}
        log.update(info)
        wandb.log(log, step=step)

    def report(self, step: int, info: Dict[str, int | float | str]):
        print(f'Step {step}; report: {info}')

        if not self.wandb_enabled:
            return
        log = info
        wandb.log(log, step=step)

    def report_sample(self, step: int, head: str, pred: Iterable[str], target: Iterable[str]):
        columns = ['Target', 'Predicted*']
        data = [[target_, pred_]
                for pred_, target_ in zip(pred, target)]
        sample_table = wandb.Table(data=data, columns=columns)
        if not self.wandb_enabled:
            return
        wandb.log({f'{head} samples': sample_table}, step=step)


@dataclass
class QAItem:
    tensor: torch.Tensor | torch.LongTensor
    type: int

    def to(self, device):
        self.tensor = self.tensor.to(device)
        return self


@dataclass
class QASequence:
    sequence: List[QAItem]

    def to(self, device):
        for i in range(len(self.sequence)):
            self.sequence[i] = self.sequence[i].to(device)
        return self


@dataclass
class QABatch:
    items: List[QASequence]

    def to(self, device):
        for i in range(len(self.items)):
            self.items[i] = self.items[i].to(device)
        return self


class LibriQwenDataset(Dataset):
    def __init__(self,
                 dataset: LibriLogMel,
                 tokenizer: AutoTokenizer,
                 eos_token: int = EOS_TOKEN,
                 eoa_token: int = EOA_TOKEN):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.eos_token = eos_token
        self.eoa_token = eoa_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, id) -> QASequence:
        item = self.dataset[id]
        log_mel = item['log-mel']
        tokens = self.tokenizer(
            item['text'], add_special_tokens=False, return_tensors='pt').input_ids.flatten()
        tokens = torch.concatenate((
            torch.tensor([self.eoa_token]),
            tokens,
            torch.tensor([self.eos_token]))
        )
        return QASequence([
            QAItem(log_mel, 1),
            QAItem(tokens, 0)
        ])


class QwenWithAdapter:
    mask_token = -100
    adapter_fname = 'adapter.pt'
    ext_emb_fname = 'ext_emb.pt'
    qwen_fname = 'qwen.pt'

    def __init__(self,
                 adapter: RawMelAdapter,
                 ext_emb: nn.Module,
                 qwen: AutoModelForCausalLM,
                 ext_base: int = EXT_TOKEN_BASE,
                 device=device
                 ):
        self.adapter = adapter
        self.ext_emb = ext_emb
        self.qwen = qwen
        self.qwen_emb = qwen.model.embed_tokens

        self.ext_base = ext_base
        self.device = device

        self.emb_dim = self.qwen_emb.embedding_dim

    def freeze_ext_emb(self):
        for parameter in self.ext_emb.parameters():
            parameter.requires_grad = False

    def freeze_qwen(self):
        for parameter in self.qwen.model.parameters():
            parameter.requires_grad = False

    def freeze_adapter(self):
        for parameter in self.adapter.parameters():
            parameter.requires_grad = False

    def parameters(self):
        raw_params = list(self.adapter.parameters()) +\
            list(self.qwen.model.parameters()) +\
            list(self.ext_emb.parameters())
        params = list(filter(lambda param: param.requires_grad, raw_params))
        return params

    def _get_token_emb(self, tokens: torch.Tensor):
        # tokens is tensor [N_tokens]
        ext_token_mask = (tokens >= self.ext_base)
        emb = torch.zeros(
            (tokens.shape[0], self.emb_dim), requires_grad=True).to(self.device)
        emb[~ext_token_mask] = self.qwen_emb(tokens[~ext_token_mask])
        emb[ext_token_mask] = self.ext_emb(
            tokens[ext_token_mask] - self.ext_base)
        return emb

    def _get_audio_emb(self, audio: torch.Tensor):
        # audio is tensor [N_channels x M_frames]
        audio = audio.unsqueeze(0)
        emb = self.adapter(audio)
        return emb.squeeze(0)

    def _get_emb(self, qa_seq: QASequence):
        embeddings = []
        labels = []

        for qa_item in qa_seq.sequence:
            if qa_item.type == 0:  # tokens
                token_emb = self._get_token_emb(qa_item.tensor)
                token_label = qa_item.tensor.detach().clone()
                token_label[token_label >= self.ext_base] = -100
                embeddings.append(token_emb)
                labels.append(token_label)
            else:
                audio_emb = self._get_audio_emb(qa_item.tensor)
                audio_label = torch.full(
                    (audio_emb.shape[0],), self.mask_token).to(self.device)
                embeddings.append(audio_emb)
                labels.append(audio_label)

        embeddings = torch.concatenate(embeddings, dim=0)
        labels = torch.concatenate(labels, dim=0)
        return embeddings, labels

    def _pad_emb_label(self,
                       iemb: List[torch.Tensor],
                       ilabels: List[torch.LongTensor]):
        max_len = max([emb.shape[0] for emb in iemb])

        pad_emb = []
        pad_labels = []

        for emb, label in zip(iemb, ilabels):
            pad_shape = max_len - emb.shape[0]
            pad_emb.append(
                F.pad(emb, (0, 0, 0, pad_shape)).unsqueeze(0))
            pad_labels.append(
                F.pad(label, (0, pad_shape), value=self.mask_token).unsqueeze(0))
        pad_emb = torch.concatenate(pad_emb, dim=0)
        pad_labels = torch.concatenate(pad_labels, dim=0)
        return pad_emb, pad_labels

    def _get_batch_emb(self, batch: QABatch):
        batch_emb = []
        batch_labels = []

        for sa_seq in batch.items:
            emb, labels = self._get_emb(sa_seq)
            batch_emb.append(emb)
            batch_labels.append(labels)

        return self._pad_emb_label(batch_emb, batch_labels)

    # https://huggingface.co/docs/transformers/en/model_doc/qwen2?usage=AutoModel#transformers.Qwen2Model.forward
    def forward(self, batch: QABatch, **kwargs):
        embeddings, labels = self._get_batch_emb(batch)
        out = self.qwen(
            inputs_embeds=embeddings,
            labels=labels,
            **kwargs
        )
        return out, labels

    def generate(self,
                 items: Iterable[Tuple[torch.Tensor, int]],
                 **kwargs):
        embeddings = self._get_emb(items)
        res = self.qwen.generate(inputs_embeds=embeddings, **kwargs)
        return res

    def to(self, device):
        self.ext_emb = self.ext_emb.to(device)
        self.adapter = self.adapter.to(device)
        self.qwen = self.qwen.to(device)
        self.qwen_emb = self.qwen_emb.to(device)
        return self

    def train(self):
        self.ext_emb.train()
        self.adapter.train()
        self.qwen.model.train()

    def eval(self):
        self.ext_base.eval()
        self.adapter.eval()
        self.qwen.model.eval()

    @classmethod
    def init(clf, adapter_config, ext_emb_config, device):
        adapter = RawMelAdapter(**adapter_config).to(device)
        ext_emb = nn.Embedding(**ext_emb_config).to(device)
        qwen = AutoModelForCausalLM.from_pretrained(QWEN_REPO).to(device)
        return clf(adapter, ext_emb, qwen, device=device)

    def save(self, dir: str):
        torch.save(self.adapter.state_dict(),
                   os.path.join(dir, self.adapter_fname))
        torch.save(self.ext_emb.state_dict(),
                   os.path.join(dir, self.ext_emb_fname))
        torch.save(self.qwen.model.state_dict(),
                   os.path.join(dir, self.qwen_fname))

    def load(self, dir: str):
        self.adapter.load_state_dict(torch.load(
            os.path.join(dir, self.adapter_fname),
            weights_only=True
        ))
        self.ext_emb.load_state_dict(torch.load(
            os.path.join(dir, self.ext_emb_fname),
            weights_only=True
        ))
        self.qwen.model.load_state_dict(torch.load(
            os.path.join(dir, self.qwen_fname),
            weights_only=True
        ))


class Trainer:
    def __init__(self,
                 config,
                 train_ds: LibriQwenDataset,
                 val_ds: LibriQwenDataset,
                 tokenizer: AutoTokenizer,
                 model: QwenWithAdapter,
                 opt,
                 scheduler,
                 reporter: Reporter,
                 device=device):
        self.config = config
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.tokenizer = tokenizer
        self.model = model
        self.opt = opt
        self.scheduler = scheduler
        self.reporter = reporter
        self.device = device
        self.best_metric = None
        self._setup_dataloaders()

    def _setup_dataloaders(self):
        self.train_dataloader = DataLoader(
            self.train_ds,
            batch_size=self.config.train.batch_size,
            num_workers=self.config.train.dl_workers,
            prefetch_factor=self.config.train.prefetch_factor,
            collate_fn=Trainer.collate_fn,
            shuffle=True
        )
        self.val_dataloader = DataLoader(
            self.val_ds,
            batch_size=self.config.train.batch_size,
            num_workers=self.config.train.dl_workers,
            prefetch_factor=self.config.train.prefetch_factor,
            collate_fn=Trainer.collate_fn,
            shuffle=False
        )

    @torch.no_grad()
    def _train_step_callback(self, step: int, loss: float):
        if step % self.config.train.val_freq == 0:
            val_info = self._calc_val_metrics()
            train_info = self._calc_train_metrics()
            self.reporter.report(step, val_info)
            self.reporter.report(step, train_info)

            if self.best_metric is None or ('Val WER*' in val_info and val_info['Val WER*'] < self.best_metric):
                self.best_metric = val_info['Val WER*']
                self._best_checkpoint()

        if step % self.config.train.save_freq == 0:
            self._checkpoint()

        if step % self.config.log.sample_freq == 0:
            self._log_train_sample(step)
            self._log_val_sample(step)

    def _checkpoint(self):
        print('Saving new checkpoint')
        self.model.save(self.config.train.checkpoint)

    def _best_checkpoint(self):
        print('Saving new best checkpoint')
        self.model.save(self.config.train.best_checkpoint)

    @torch.no_grad()
    def _calc_train_metrics(self) -> Dict[str, Any]:
        pred = []
        target = []
        total = 0
        for batch in self.train_dataloader:
            if total >= self.config.train.train_metric_subset:
                break
            total += len(batch.items)

            batch = batch.to(self.device)
            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                out, labels = self.model.forward(batch)
            pred_batch, target_batch = self._get_pred_target(out, labels)
            pred.extend(pred_batch)
            target.extend(target_batch)

        pred = self._filter_empty(pred)
        target = self._filter_empty(target)
        try:
            wer = calculate_wer(pred, target)
            return {'Train WER*': wer}
        except Exception as e:
            print(f'[ERR] Failed to calculate val WER because of {e}')
        return {}

    @torch.no_grad()
    def _calc_val_metrics(self) -> Dict[str, Any]:
        pred = []
        target = []
        for batch in self.val_dataloader:
            batch = batch.to(self.device)
            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                out, labels = self.model.forward(batch)
            pred_batch, target_batch = self._get_pred_target(out, labels)
            pred.extend(pred_batch)
            target.extend(target_batch)
        pred = self._filter_empty(pred)
        target = self._filter_empty(target)
        try:
            wer = calculate_wer(pred, target)
            return {'Val WER*': wer}
        except Exception as e:
            print(f'[ERR] Failed to calculate train WER because of {e}')
        return {}

    def _log_train_sample(self, step: int):
        pred, target = self._get_sample(self.train_dataloader)
        self.reporter.report_sample(step, 'Train', pred, target)

    def _log_val_sample(self, step: int):
        pred, target = self._get_sample(self.val_dataloader)
        self.reporter.report_sample(step, 'Val', pred, target)

    @staticmethod
    def _filter_empty(items: Iterable[str]):
        return list(filter(lambda s: len(s) != 0, items))

    def _get_sample(self, dataloader):
        pred = []
        target = []
        total = 0

        for batch in dataloader:
            if total >= self.config.log.sample_size:
                break
            total += len(batch.items)

            batch = batch.to(self.device)
            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                out, labels = self.model.forward(batch)
            pred_batch, target_batch = self._get_pred_target(out, labels)
            pred.extend(pred_batch)
            target.extend(target_batch)
        return pred, target

    def _get_pred_target(self, out, labels):
        pred = []
        target = []

        batch_size = labels.shape[0]
        for i in range(batch_size):
            ilogits = out.logits[i]
            ilabels = labels[i]

            mask = (ilabels != self.model.mask_token) &\
                (ilabels < self.model.ext_base)
            ids = torch.where(mask)[0]

            pred_tokens = torch.argmax(ilogits[ids - 1], dim=-1)
            target_tokens = ilabels[mask]

            pred.append(self.tokenizer.decode(pred_tokens))
            target.append(self.tokenizer.decode(target_tokens))

        return pred, target

    def _opt_step(self) -> float:
        grad_norm = calculate_grad_norm(self.model)

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.train.grad_clip
        )

        self.opt.step()
        self.scheduler.step()

        self.opt.zero_grad()
        return {'grad_norm': grad_norm}

    def train(self):
        batch_step = 0
        global_step = 0
        loss = 0

        for epoch in range(self.config.train.epoches):
            print(f'Start epoch {epoch}')

            self.model.train()
            for i, batch in tqdm(enumerate(self.train_dataloader)):
                batch = batch.to(device)
                with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                    output, _ = self.model.forward(batch)
                    batch_loss = output.loss

                batch_loss *= 1 / self.config.train.gradient_accumulation_steps
                batch_loss.backward()
                batch_step += 1
                loss += batch_loss.item()

                if batch_step == self.config.train.gradient_accumulation_steps:
                    info = self._opt_step()
                    info['epoch'] = epoch + i / len(self.train_dataloader)
                    self.reporter.report_train_loss(global_step, loss, info)

                    self._train_step_callback(global_step, loss)
                    loss = 0
                    batch_step = 0
                    global_step += 1

                torch.cuda.empty_cache()
                gc.collect()

    @staticmethod
    def collate_fn(batch) -> tuple[torch.Tensor, torch.Tensor]:
        return QABatch(list(batch))


def main():
    parser = argparse.ArgumentParser(
        prog='Mel-Qwen training script')
    parser.add_argument('--config', required=True, type=str)
    args = parser.parse_args()

    config = read_config(args.config)

    print('Loading dataset')
    tokenizer = load_qwen_tokenizer()
    train_raw_ds = [
        LIBRISPEECH(
            root=config.dataset.train.root[i],
            url=config.dataset.train.split[i],
            download=False)
        for i in range(len(config.dataset.train.root))
    ]
    train_dataset = LibriQwenDataset(
        LibriLogMel(LibriDataset(MergeDataset(train_raw_ds))),
        tokenizer)

    test_raw_ds = [
        LIBRISPEECH(
            root=config.dataset.test.root[i],
            url=config.dataset.test.split[i],
            download=False)
        for i in range(len(config.dataset.test.root))
    ]
    test_dataset = LibriQwenDataset(
        LibriLogMel(LibriDataset(MergeDataset(test_raw_ds))),
        tokenizer)

    val_dataset = Subset(
        test_dataset,
        torch.randint(len(test_dataset), size=(
            config.dataset.test.val_size,)).tolist()
    )

    print("Loading model")
    model = QwenWithAdapter.init(
        config.model.adapter_config,
        config.model.ext_emb_config,
        device=device
    )
    if hasattr(config.model, "checkpoint"):
        print('Loading checkpoint')
        model.load(config.model.checkpoint)
    elif hasattr(config.model, "adapter_checkpoint"):
        model.adapter.load_state_dict(torch.load(
            config.model.adapter_checkpoint, weights_only=True))

    if config.train.freeze_qwen:
        print("Freezing QWEN")
        model.freeze_qwen()
    if config.train.freeze_adapter:
        print("Freezing adapter")
        model.freeze_adapter()

    opt = torch.optim.AdamW(model.parameters(), config.opt.lr)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        opt,
        start_factor=0.001,
        end_factor=1.0,
        total_iters=config.opt.warmup_steps
    )

    print('Setting up reporting')
    reporter = Reporter(config, config.wandb.enabled)
    reporter.setup()

    print('Start training')
    trainer = Trainer(
        config,
        train_dataset,
        val_dataset,
        tokenizer,
        model,
        opt,
        warmup_scheduler,
        reporter,
        device
    )

    try:
        trainer.train()
    finally:
        wandb.finish()


if __name__ == '__main__':
    main()
