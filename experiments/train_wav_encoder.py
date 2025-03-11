import argparse
import copy
import json
import os
from typing import List

import datasets
import torch
import torch.functional as F
import torch.nn as nn
from datasets import load_from_disk
from munch import Munch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb

from StreamASRLib.model import WavTokensEncoder

device = 'cuda' if torch.cuda.is_available() else 'cpu'

WANDB_TOKEN = os.environ['WANDB_TOKEN']

QWEN_REPO = 'Qwen/Qwen2.5-0.5B'
QWEN_EMB_SIZE = 896

EOS_TOKEN = 151643
TEXT_PADDING_TOKEN = EOS_TOKEN

WAV_TOKENS = 4096
BOA_TOKEN = WAV_TOKENS
EOA_TOKEN = BOA_TOKEN + 1
AUDIO_PADDING_TOKEN = EOA_TOKEN
AUDIO_TOKENS = WAV_TOKENS + 2


def colBertScore(audio_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
    '''
        audio_emb has shape (batch_size, audio_seq_len, emb_dim)

        text_emb has shape (batch_size, text_seq_len, emb_dim)

        Returns tensor of col bert score of shape (batch_size,)
    '''
    text_emb_T = text_emb.transpose(-1, -2)
    max_similarity, _ = (audio_emb @ text_emb_T).max(dim=-1)

    # In original col bert, the sum of max similirarity is taken instead of mean
    return max_similarity.sum(dim=-1)


def colBertTripletLoss(audio_emb: torch.Tensor, text_pos_emb: torch.Tensor, text_neg_emb: torch.Tensor) -> torch.Tensor:
    '''
        audio_emb has shape (batch_size, audio_seq_len, emb_dim)

        text_pos_emb has shape (batch_size, text_pos_seq_len, emb_dim)

        text_pos_emb has shape (batch_size, text_pos_seq_len, emb_dim)

        Returns loss of the batch
    '''
    pos_score = colBertScore(audio_emb, text_pos_emb)
    neg_score = colBertScore(audio_emb, text_neg_emb)
    logits = torch.concatenate((neg_score, pos_score), dim=-1)
    loss = nn.CrossEntropyLoss()(logits, torch.tensor(1).to(logits.device))
    return loss


def read_config(path: str):
    with open(path, "r") as config_file:
        return Munch.fromDict(json.load(config_file))


def load_qwen_emb() -> nn.Embedding:
    qwen_model = AutoModelForCausalLM.from_pretrained(QWEN_REPO)
    qwen_emb = copy.deepcopy(qwen_model.model.embed_tokens)
    del qwen_model
    return qwen_emb


def load_qwen_tokenizer(use_fast: bool = True):
    return AutoTokenizer.from_pretrained(QWEN_REPO, use_fast=use_fast)


@torch.no_grad()
def get_qwen_emb(qwen_emb, tokens):
    return qwen_emb(tokens)


def report_loss(step: int, loss: float):
    log = {'train_loss': loss}
    wandb.log(log, step=step)


def report_sample(
    pos_text: List[str],
    neg_text: List[str],
    pos_score: List[float],
    neg_score: List[float],
    num_samples: int = 10
):
    pos_samples = pos_text[:num_samples]
    neg_samples = neg_text[:num_samples]
    pos_scores = pos_score[:num_samples]
    neg_scores = neg_score[:num_samples]

    data = [[pos_sample, neg_sample, pos_score, neg_score]
            for pos_sample, neg_sample, pos_score, neg_score in
            zip(pos_samples, neg_samples, pos_scores, neg_scores)]
    columns = ['Positive Sample', 'Negative Sample',
               'Positive Score', 'Negative Score']
    print(data)
    sample_table = wandb.Table(data=data, columns=columns)
    wandb.log({'Sample Table': sample_table})


class DummyTripletSampler:
    class TripletDatasetWrapper(Dataset):
        def __init__(self, p_ds, n_ds):
            '''
            p_ds, n_ds return dict with keys "wav-tokens" and "qwen-tokens"
            '''
            super().__init__()
            self.p_ds = p_ds
            self.n_ds = n_ds

        def __len__(self):
            return len(self.p_ds)

        def __getitem__(self, item):
            wav_tokens = self.p_ds[item]['wav-tokens']
            pos_tokens = self.p_ds[item]['qwen-tokens']
            neg_tokens = self.n_ds[item]['qwen-tokens']
            return (torch.tensor([BOA_TOKEN] + wav_tokens + [EOA_TOKEN]), torch.tensor(pos_tokens), torch.tensor(neg_tokens))

    def __init__(self, dataset, sample_size: int = 50_000):
        self.dataset = dataset
        self.sample_size = sample_size

    def _sample_from_ds(self, size):
        ids = torch.randint(len(self.dataset), size=(size,))
        sample_ds = self.dataset.select(ids)
        return sample_ds

    def generate_new_dataset(self):
        p_ds = self._sample_from_ds(self.sample_size)
        n_ds = self._sample_from_ds(self.sample_size)
        return DummyTripletSampler.TripletDatasetWrapper(p_ds, n_ds)


def tokens_padding(tensors: List[torch.Tensor], pad_token: int) -> torch.Tensor:
    max_length = max(item.shape[0] for item in tensors)
    pad_tensor = torch.full((len(tensors), max_length), pad_token)
    for i, item in enumerate(tensors):
        pad_tensor[i, :item.shape[0]] = item
    return pad_tensor


def triplet_collate_fn(batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global TEXT_PADDING_TOKEN, AUDIO_PADDING_TOKEN
    pad_wav = tokens_padding([item[0] for item in batch], AUDIO_PADDING_TOKEN)
    pad_pos = tokens_padding([item[1] for item in batch], TEXT_PADDING_TOKEN)
    pad_neg = tokens_padding([item[2] for item in batch], TEXT_PADDING_TOKEN)
    return (pad_wav, pad_pos, pad_neg)


def main():
    parser = argparse.ArgumentParser(prog='Wav-Tokens-Encoder training script')
    parser.add_argument('--config', required=True, type=str)
    args = parser.parse_args()

    config = read_config(args.config)

    print('Loading dataset')
    train_dataset = load_from_disk(config.dataset.librispeech_train)

    print('Loading model')
    tokenizer = load_qwen_tokenizer()
    qwen_emb = load_qwen_emb()
    if hasattr(config.model, 'load_checkpoint'):
        model = WavTokensEncoder.load(
            config.model.load_checkpoint, AUDIO_TOKENS, QWEN_EMB_SIZE, config.model.num_layers)
    else:
        model = WavTokensEncoder(
            AUDIO_TOKENS, QWEN_EMB_SIZE, config.model.num_layers, add_rope=False)
    model = model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), config.opt.lr)

    print('Initialising W&B')
    wandb.login(key=WANDB_TOKEN)
    wandb.init(
        project=config.wandb.project,
        config={},
        name=config.wandb.run_name,
        # disable system logging
        settings=wandb.Settings(_disable_stats=True, _disable_meta=True)
    )

    print('Start training')
    triplet_sampler = DummyTripletSampler(train_dataset)
    step = 0
    for epoch in range(config.train.epoches):
        print(f'Start epoch {epoch}')
        triplet_ds = triplet_sampler.generate_new_dataset()
        triplet_dataloader = DataLoader(
            triplet_ds,
            batch_size=config.train.batch_size,
            collate_fn=triplet_collate_fn,
            shuffle=True,
            num_workers=config.train.dl_workers
        )

        loss = 0
        last_loss = None
        best_loss = None
        accumulated = 0

        model.train()
        opt.zero_grad()
        for batch in tqdm(triplet_dataloader):
            wav_emb = model(batch[0].to(device))
            text_pos_emb = get_qwen_emb(qwen_emb, batch[1]).to(device)
            text_neg_emb = get_qwen_emb(qwen_emb, batch[2]).to(device)

            batch_loss = colBertTripletLoss(
                wav_emb, text_pos_emb, text_neg_emb) / config.train.gradient_accumulation_steps
            batch_loss.backward()

            loss += batch_loss.item()
            accumulated += 1
            step += 1

            if accumulated == config.train.gradient_accumulation_steps:
                opt.step()
                opt.zero_grad()

                accumulated = 0
                print(f'Step {step} loss: {loss}')
                report_loss(
                    step // config.train.gradient_accumulation_steps, loss)
                last_loss = loss
                loss = 0

            if step > 0 and step % config.log.sample_freq == 0:
                pos_score = colBertScore(wav_emb, text_pos_emb)
                neg_score = colBertScore(wav_emb, text_neg_emb)
                pos_text = [tokenizer.decode(item) for item in batch[1]]
                neg_text = [tokenizer.decode(item) for item in batch[2]]
                report_sample(pos_text, neg_text, pos_score, neg_score)

            if step > 0 and step % config.train.save_freq == 0:
                model.save(config.train.save_checkpoint_path)
                if best_loss is None or best_loss > last_loss:
                    model.save(config.train.best_checkpoint_path)


if __name__ == '__main__':
    main()
