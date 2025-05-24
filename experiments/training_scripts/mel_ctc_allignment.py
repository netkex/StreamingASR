import argparse
import copy
import gc
import os
from typing import Dict, List, Tuple

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

from torchtune.modules import RotaryPositionalEmbeddings

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)

WANDB_TOKEN = os.environ['WANDB_TOKEN']

QWEN_REPO = 'Qwen/Qwen2.5-0.5B'
QWEN_EMB_SIZE = 896

EOS_TOKEN = 151643
TEXT_PADDING_TOKEN = EOS_TOKEN


class BlankEmb(nn.Module):
    def __init__(self, emb_dim: int = QWEN_EMB_SIZE):
        super().__init__()
        self.blank_emb_ = nn.Parameter(
            F.normalize(torch.randn(emb_dim, requires_grad=True), dim=-1))

    def blank_emb(self) -> torch.Tensor:
        return self.blank_emb_

    def save(self, path: str):
        torch.save(self.state_dict(), path)


class CTC:
    '''
        The CTC is used to calculate CTC loss and CTC allignment of audio embeddings.

        Shapes:

        * aduio_emb is tensor of audio embeddings of size (B, L_w, E)
        where B is size of batch; L_w is padded length; E - size of embedding.

        * target is tensor of original (unmapped) qwen tokens of size (B, L_t)
        where B is size of batch; L_t is padded length.

        * audio_length is tensor with input lengths of size (B,)
        where B is size of batch.

        * target_length is tensor with text lengths of size (B,)
        where B is size of batch.

        All tensors must be on the same device.
    '''

    def __init__(self,
                 qwen_embeddings: torch.Tensor,
                 qwen_tokens_mapping: torch.Tensor,
                 blank_emb: BlankEmb,
                 temp: float = 0.01,
                 device=device):
        self.blank_id = qwen_embeddings.shape[0]
        self.ctc_loss = nn.CTCLoss(blank=self.blank_id)
        self.qwen_embeddings = qwen_embeddings.to(device)
        self.qwen_tokens_mapping = qwen_tokens_mapping.to(device)
        self.blank_emb = blank_emb
        self.temp = temp

    # We need to do log_softmax in fp32 for numerical stability
    # https://discuss.pytorch.org/t/ctc-loss-ctc-loss-not-support-float16/148800/2
    def _get_probs_logits(self, audio_emb: torch.Tensor) -> torch.Tensor:
        qwen_embedding_matrix = torch.concatenate(
            (self.qwen_embeddings, self.blank_emb.blank_emb().reshape((1, -1))))
        qwen_embedding_matrix = F.normalize(qwen_embedding_matrix, dim=-1)
        audio_emb = F.normalize(audio_emb, dim=-1)

        raw_logits = (audio_emb @ qwen_embedding_matrix.T) / self.temp
        raw_logits = F.log_softmax(raw_logits, dim=-1)
        return raw_logits

    def _map_target(self, target: torch.Tensor) -> torch.Tensor:
        return self.qwen_tokens_mapping[target]

    def loss(self,
             audio_emb: torch.Tensor,
             target: torch.Tensor,
             audio_length: torch.Tensor,
             target_length: torch.Tensor):

        mapped_target = self._map_target(target)
        logits = self._get_probs_logits(audio_emb)
        logits = logits.transpose(0, 1)

        assert (torch.all(mapped_target != self.blank_id))
        loss = self.ctc_loss(logits, mapped_target,
                             audio_length, target_length)
        return loss

    def allignment_matrix(self,
                          audio_emb: torch.Tensor,
                          target: torch.Tensor):
        mapped_target = self._map_target(target)
        logits = self._get_probs_logits(audio_emb)
        probs = torch.exp(logits)

        target_probs = torch.zeros(
            (audio_emb.shape[0], audio_emb.shape[1], target.shape[1]))
        for i in range(target.shape[0]):
            target_probs[i, :, :] = probs[i, :, mapped_target[i]]
        return target_probs


class Reporter:
    def __init__(self, config, wandb_enabled: bool = True):
        self.config = config
        self.wandb_enabled = wandb_enabled
        pass

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

    def report_train_loss(self,
                          step: int,
                          loss: float,
                          info: Dict[str, int | float | str] = None):
        if info is None:
            info = {}

        print(f'Step {step}; train loss: {loss}; info: {info}')
        if not self.wandb_enabled:
            return
        log = {'train_loss': loss}
        for k, v in info.items():
            log[k] = v
        wandb.log(log, step=step)

    def report_val_loss(self, step: int, loss: float):
        print(f'Step {step}; val loss: {loss}')
        if not self.wandb_enabled:
            return
        log = {'val_loss': loss}
        wandb.log(log, step=step)

    def report_alligment_matrix(self,
                                step: int,
                                head: str,  # train or validation
                                alligment_matrix: List[torch.Tensor],
                                labels: List[List[str]]):
        if not self.wandb_enabled:
            return
        log = {}
        for i in range(len(labels)):
            plt = px.imshow(
                alligment_matrix[i].detach().cpu(),
                x=labels[i],
                y=list(range(alligment_matrix[i].shape[0]))
            )
            log[f'{head} CTC matrix {i}'] = plt
        wandb.log(log, step=step)


class LibriQwenDataset(Dataset):
    def __init__(self,
                 dataset: LibriLogMel,
                 tokenizer: AutoTokenizer):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, id):
        item = self.dataset[id]
        log_mel = item['log-mel']
        qwen_tokens = self.tokenizer(
            item['text'], add_special_tokens=False, return_tensors='pt').input_ids.flatten()
        return (log_mel, qwen_tokens)


def calculate_grad_norm(model):
    return torch.sqrt(sum([torch.norm(param.grad) ** 2 for param in model.parameters()]))


def get_number_of_parameters(model):
    return sum(p.numel() for p in model.parameters())


class Trainer:
    def __init__(self,
                 config,
                 train_ds: LibriQwenDataset,
                 val_ds: LibriQwenDataset,
                 tokenizer: AutoTokenizer,
                 adapter: RawMelAdapter,
                 blank_emb: BlankEmb,
                 opt,
                 scheduler,
                 ctc: CTC,
                 reporter: Reporter):
        self.config = config
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.tokenizer = tokenizer
        self.adapter = adapter
        self.blank_emb = blank_emb
        self.opt = opt
        self.scheduler = scheduler
        self.ctc = ctc
        self.reporter = reporter
        self.best_loss = None
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
    def _train_step_callback(self,
                             step: int,
                             loss: float,
                             last_audio_emb: torch.Tensor,
                             last_audio_length: torch.Tensor,
                             last_batch):
        if step % self.config.train.val_freq == 0:
            loss = self._validate()
            self.reporter.report_val_loss(step, loss)

        if step % self.config.train.save_freq == 0:
            self._save_models()
            if self.best_loss is None or loss < self.best_loss:
                self._save_best()
                self.best_loss = loss

        if step % self.config.log.sample_freq == 0:
            self._log_train_sample(step, last_audio_emb,
                                   last_audio_length, last_batch)
            self._log_val_sample(step)

    def _save_models(self):
        self.adapter.save(self.config.train.adapter_checkpoint)
        self.blank_emb.save(self.config.train.blank_checkpoint)

    def _save_best(self):
        self.adapter.save(self.config.train.adapter_best_checkpoint)
        self.blank_emb.save(self.config.train.blank_best_checkpoint)

    @torch.no_grad()
    def _validate(self) -> float:
        loss = 0
        self.adapter.eval()
        self.blank_emb.eval()
        for batch in tqdm(self.val_dataloader):
            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                audio_emb = self.adapter(batch[0].to(device))
                audio_length = self.adapter.get_length(batch[1])
                ctc_loss = self.ctc.loss(
                    audio_emb,
                    batch[2].to(device),
                    audio_length.to(device),
                    batch[3].to(device)
                )
            loss += ctc_loss.item()
        self.adapter.train()
        self.blank_emb.train()
        return loss / len(self.val_dataloader)

    def _create_allignment_samples(self,
                                   num_samples: int,
                                   audio_emb: torch.Tensor,
                                   target: torch.Tensor,
                                   audio_length: torch.Tensor,
                                   target_length: torch.Tensor):
        num_samples = min(num_samples, audio_emb.shape[0])
        audio_emb = audio_emb[:num_samples].float()
        target = target[:num_samples]
        audio_length = audio_length[:num_samples]
        target_length = target_length[:num_samples]

        target_with_blanks = torch.full(
            (num_samples, 2 * target.shape[1] - 1), self.ctc.blank_id)
        target_with_blanks[:, 2 * torch.arange(target.shape[1])] = target
        target_with_blanks_length = 2 * target_length - 1

        raw_allignment = self.ctc.allignment_matrix(
            audio_emb, target_with_blanks)

        alligment = []
        symbol_target = []
        for i in range(num_samples):
            audio_length_ = audio_length[i].item()
            target_length_ = target_with_blanks_length[i].item()
            alligment.append(
                raw_allignment[i, :audio_length_, :target_length_])

            symbol_target_ = []
            for j in range(target_length[i]):
                symbol_target_.append(self.tokenizer.decode(target[i, j]))
                if j != target_length[i] - 1:
                    symbol_target_.append(r'$\epsilon$')
            symbol_target.append(symbol_target_)

        return (alligment, symbol_target)

    def _log_train_sample(self,
                          step: int,
                          last_audio_emb: torch.Tensor,
                          last_audio_length: torch.Tensor,
                          last_batch):
        num_samples = self.config.log.num_samples
        allignment, target = self._create_allignment_samples(
            num_samples,
            last_audio_emb,
            last_batch[2],
            last_audio_length,
            last_batch[3]
        )
        self.reporter.report_alligment_matrix(
            step, 'train', allignment, target)

    def _log_val_sample(self, step: int):
        pass

    def _opt_step(self) -> float:
        adapter_grad_norm = calculate_grad_norm(self.adapter)
        blank_grad_norm = calculate_grad_norm(self.blank_emb)

        torch.nn.utils.clip_grad_norm_(
            list(self.adapter.parameters()) +
            list(self.blank_emb.parameters()),
            self.config.train.grad_clip
        )

        self.opt.step()
        self.scheduler.step()

        self.opt.zero_grad()
        return {'adapter_grad_norm': adapter_grad_norm, 'blank_grad_norm': blank_grad_norm}

    def train(self):
        batch_step = 0
        global_step = 0
        loss = 0

        for epoch in range(self.config.train.epoches):
            print(f'Start epoch {epoch}')

            self.adapter.train()
            self.blank_emb.train()
            for i, batch in tqdm(enumerate(self.train_dataloader)):
                with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                    audio_emb = self.adapter(batch[0].to(device))
                    audio_length = self.adapter.get_length(batch[1])
                    ctc_loss = self.ctc.loss(
                        audio_emb,
                        batch[2].to(device),
                        audio_length.to(device),
                        batch[3].to(device)
                    )

                batch_loss = 1 / self.config.train.gradient_accumulation_steps * ctc_loss
                batch_loss.backward()
                batch_step += 1
                loss += batch_loss.item()

                if batch_step == self.config.train.gradient_accumulation_steps:
                    info = self._opt_step()
                    info['epoch'] = epoch + i / len(self.train_dataloader)
                    self.reporter.report_train_loss(global_step, loss, info)

                    self._train_step_callback(
                        global_step, loss, audio_emb, audio_length, batch)
                    loss = 0
                    batch_step = 0
                    global_step += 1

                torch.cuda.empty_cache()
                gc.collect()

    @staticmethod
    def _mel_padding(tensors: List[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
        lengths = [item.shape[1] for item in tensors]
        max_length = max(lengths)
        padded_tensors = torch.full(
            (len(tensors), tensors[0].shape[0], max_length), pad_value)
        for i, item in enumerate(tensors):
            padded_tensors[i, :, :item.shape[1]] = item
        return padded_tensors, torch.tensor(lengths)

    @staticmethod
    def _tokens_padding(tensors: List[torch.Tensor], pad_token: int) -> torch.Tensor:
        lengths = [item.shape[0] for item in tensors]
        max_length = max(lengths)
        padded_tensors = torch.full((len(tensors), max_length), pad_token)
        for i, item in enumerate(tensors):
            padded_tensors[i, :item.shape[0]] = item
        return padded_tensors, torch.tensor(lengths)

    @staticmethod
    def collate_fn(batch) -> tuple[torch.Tensor, torch.Tensor]:
        global TEXT_PADDING_TOKEN
        pad_audio, audio_length = Trainer._mel_padding(
            [item[0] for item in batch])
        pad_text, text_length = Trainer._tokens_padding(
            [item[1] for item in batch], TEXT_PADDING_TOKEN)
        return (pad_audio, audio_length, pad_text, text_length)


def load_qwen_emb() -> nn.Embedding:
    qwen_model = AutoModelForCausalLM.from_pretrained(QWEN_REPO)
    qwen_emb = copy.deepcopy(qwen_model.model.embed_tokens)
    del qwen_model
    return qwen_emb


@torch.no_grad()
def get_qwen_emb(qwen_emb, tokens):
    return qwen_emb(tokens)


def load_qwen_tokenizer(use_fast: bool = True) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(QWEN_REPO, use_fast=use_fast)


def prepare_qwen_tokens_with_blank(
    qwen_emb: nn.Module,
    tokens_path: str,
    _: AutoTokenizer
) -> Tuple[torch.Tensor, torch.Tensor]:
    global QWEN_EMB_SIZE
    tokens = torch.load(tokens_path)
    embeddings = torch.zeros(
        (tokens.shape[0], QWEN_EMB_SIZE), requires_grad=True)
    with torch.no_grad():
        embeddings = qwen_emb(tokens)

    tokens_mapping = torch.zeros(tokens.max() + 1, dtype=torch.long)
    tokens_mapping[tokens] = torch.arange(tokens.shape[0])
    return (embeddings, tokens_mapping)


def main():
    parser = argparse.ArgumentParser(
        prog='WavTokens CTC Audio Adapter training script')
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

    print('Loading model')
    if hasattr(config.model, 'adapter_checkpoint'):
        raise NotImplemented('Not supported at the moment')
    else:
        adapter = RawMelAdapter(
            n_mels=80,
            hidden_size=config.model.hidden_size,
            attention_layers=config.model.num_layers,
            proj_size=QWEN_EMB_SIZE
        )
    adapter = adapter.to(device)

    if hasattr(config.model, 'blank_checkpoint'):
        raise NotImplementedError('Not supported at the moment')
    else:
        blank_emb = BlankEmb(QWEN_EMB_SIZE)
    blank_emb = blank_emb.to(device)

    opt = torch.optim.AdamW(
        list(adapter.parameters()) + list(blank_emb.parameters()),
        config.opt.lr
    )
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        opt,
        start_factor=0.001,
        end_factor=1.0,
        total_iters=config.opt.warmup_steps
    )

    print(f'Model size: {get_number_of_parameters(adapter)}')

    print('Prepairing stuff for CTC')
    qwen_emb = load_qwen_emb()
    qwen_emdebbings, tokens_mapping = prepare_qwen_tokens_with_blank(
        qwen_emb, config.dataset.qwen_tokens, tokenizer)
    ctc = CTC(qwen_emdebbings, tokens_mapping,
              blank_emb, config.model.ctc_temp)

    print('Setting up reporting')
    reporter = Reporter(config, config.wandb.enabled)
    reporter.setup()

    print('Start training')
    trainer = Trainer(
        config,
        train_dataset,
        val_dataset,
        tokenizer,
        adapter,
        blank_emb,
        opt,
        warmup_scheduler,
        ctc,
        reporter
    )

    try:
        trainer.train()
    finally:
        wandb.finish()


if __name__ == '__main__':
    main()
