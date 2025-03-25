import argparse
import copy
import gc
import os
from typing import List

import plotly.express as px
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from StreamASRLib.model import WavTokensEncoder
from StreamASRLib.training import read_config


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)

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

BLOCK_SIZE = 64


def colBertScore(
        audio_emb: torch.Tensor,
        text_emb: torch.Tensor,
        audio_mask: torch.Tensor = None,
        text_mask: torch.Tensor = None
) -> torch.Tensor:
    '''
        audio_emb has shape (batch_size, audio_seq_len, emb_dim)

        text_emb has shape (batch_size, text_seq_len, emb_dim)

        audio_mask has shape (batch_size, audio_seq_len) — mask of real audio tokens (without padding)

        text_mask has shape (batch_size, text_seq_len) - mask of real text tokens (without padding)

        Returns tensor of col bert score of shape (batch_size,)
    '''

    audio_emb = F.normalize(audio_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)

    text_emb_T = text_emb.transpose(-1, -2)
    similarity = audio_emb @ text_emb_T

    # TODO: rewrite
    if text_mask is not None:
        text_coef = torch.zeros_like(text_mask).to(similarity.device)
        text_coef[~text_mask] = -torch.inf
        similarity = similarity + text_coef[:, None, :]

    # max_similarity: (batch_size, audio_seq_len)
    max_similarity, _ = similarity.max(dim=-1)
    if audio_mask is not None:
        max_similarity[~audio_mask.to(max_similarity.device)] = 0

    # In original col bert, the sum of max similirarity is taken instead of mean
    return max_similarity.sum(dim=-1)


def colBertMatrixScore(
    audio_emb: torch.Tensor,
    text_emb: torch.Tensor,
    audio_mask: torch.Tensor = None,
    text_mask: torch.Tensor = None
) -> torch.Tensor:
    '''
        audio_emb has shape (batch_size, audio_seq_len, emb_dim)

        text_emb has shape (batch_size, text_seq_len, emb_dim)

        audio_mask has shape (batch_size, audio_seq_len) — mask of real audio tokens (without padding)

        text_mask has shape (batch_size, text_seq_len) - mask of real text tokens (without padding)

        Returns tensor of col bert matrix score of shape (batch_size, batch_size)
    '''
    audio_emb = F.normalize(audio_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)
    text_emb_T = text_emb.transpose(-1, -2)

    similarity_matrix = audio_emb[:, None, ...] @ text_emb_T[None, :, ...]

    if text_mask is not None:
        broadcasted_mask = torch.ones_like(
            similarity_matrix, dtype=torch.bool).to(similarity_matrix.device)
        broadcasted_mask = (
            broadcasted_mask & text_mask[None, :, None, :].to(similarity_matrix.device))
        similarity_matrix[~broadcasted_mask] = -torch.inf

    max_similarity_matrix, _ = similarity_matrix.max(dim=-1)
    if text_mask is not None:
        audio_coef = torch.ones_like(audio_mask).to(
            max_similarity_matrix.device)
        audio_coef[~audio_mask] = 0
        max_similarity_matrix = max_similarity_matrix * audio_coef[:, None, :]
    return max_similarity_matrix.sum(dim=-1)


def colBertMatrixBlockLoss(
        audio_emb: torch.Tensor,
        text_emb: torch.Tensor,
        audio_mask: torch.Tensor = None,
        text_mask: torch.Tensor = None,
) -> torch.Tensor:
    '''
        audio_emb has shape (batch_size, audio_seq_len, emb_dim)

        text_pos_emb has shape (batch_size, text_pos_seq_len, emb_dim)

        text_pos_emb has shape (batch_size, text_pos_seq_len, emb_dim)

        Returns loss of the batch
    '''
    wav_matrix_score = torch.zeros((audio_emb.shape[0], text_emb.shape[0]))
    for i in range(0, audio_emb.shape[0], BLOCK_SIZE):
        for j in range(0, text_emb.shape[0], BLOCK_SIZE):
            block_score = colBertMatrixScore(
                audio_emb[i:i + BLOCK_SIZE, ...],
                text_emb[j:j + BLOCK_SIZE, ...],
                audio_mask[i:i + BLOCK_SIZE, ...],
                text_mask[j:j + BLOCK_SIZE, ...]
            )
            wav_matrix_score[i:i + BLOCK_SIZE, j:j + BLOCK_SIZE] = block_score
    target = torch.arange(wav_matrix_score.shape[0]).to(
        wav_matrix_score.device)
    cross_entropy = nn.CrossEntropyLoss()
    loss = cross_entropy(wav_matrix_score, target)
    return loss, wav_matrix_score


def colBertMatrixLoss(
        audio_emb: torch.Tensor,
        text_emb: torch.Tensor,
        audio_mask: torch.Tensor = None,
        text_mask: torch.Tensor = None,
) -> torch.Tensor:
    '''
        audio_emb has shape (batch_size, audio_seq_len, emb_dim)

        text_pos_emb has shape (batch_size, text_pos_seq_len, emb_dim)

        text_pos_emb has shape (batch_size, text_pos_seq_len, emb_dim)

        Returns loss of the batch
    '''
    wav_matrix_score = colBertMatrixScore(
        audio_emb, text_emb, audio_mask, text_mask)
    text_matrix_score = colBertMatrixScore(
        text_emb, audio_emb, text_mask, audio_mask)
    target = torch.arange(wav_matrix_score.shape[0]).to(
        wav_matrix_score.device)
    cross_entropy = nn.CrossEntropyLoss()
    # loss = cross_entropy(wav_matrix_score, target)
    loss = (cross_entropy(wav_matrix_score, target) +
            cross_entropy(text_matrix_score, target)) / 2
    return loss, wav_matrix_score


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


def report_loss(step: int, loss: float, grad_norm: float):
    log = {'train_loss': loss, 'grad_norm': grad_norm}
    wandb.log(log, step=step)


def report_sample(
    texts: List[str],
    scores: torch.Tensor,
    num_samples: int = 10
):
    sample_texts = [[text] for text in texts[:num_samples]]
    text_table = wandb.Table(data=sample_texts, columns=['sentences'])

    sample_scores = scores[:num_samples, :num_samples].detach().cpu()
    sample_scores = torch.softmax(sample_scores, dim=-1)

    plt = px.imshow(sample_scores, x=list(range(num_samples)),
                    y=list(range(num_samples)), text_auto=True)
    wandb.log({'Sample texts': text_table, 'Score Softmax Heatmap': plt})


class LimitedDataset(Dataset):
    def __init__(self, dataset, max_len_wav: int = 1024, max_len_text: int = 512):
        super().__init__()
        self.dataset = dataset
        self.max_len_wav = max_len_wav
        self.max_len_text = max_len_text
        self.indexes = LimitedDataset._build_indexes(
            dataset, max_len_wav, max_len_text)

    @staticmethod
    def _build_indexes(dataset, max_len_wav, max_len_text):
        indexes = []
        for id in tqdm(range(len(dataset))):
            item = dataset[id]
            if len(item['wav-tokens']) > max_len_wav or len(item['qwen-tokens']) > max_len_text:
                continue
            indexes.append(id)
        return indexes

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, id):
        item = self.dataset[self.indexes[id]]
        wav_tokens = item['wav-tokens']
        text_tokens = item['qwen-tokens']
        return (
            torch.tensor([BOA_TOKEN] + wav_tokens + [EOA_TOKEN]
                         ), torch.tensor(text_tokens)
        )


def tokens_padding(tensors: List[torch.Tensor], pad_token: int) -> torch.Tensor:
    max_length = max([item.shape[0] for item in tensors])
    pad_tensor = torch.full((len(tensors), max_length), pad_token)
    for i, item in enumerate(tensors):
        pad_tensor[i, :item.shape[0]] = item
    return pad_tensor


def collate_fn(batch) -> tuple[torch.Tensor, torch.Tensor]:
    global TEXT_PADDING_TOKEN, AUDIO_PADDING_TOKEN
    pad_wav = tokens_padding([item[0] for item in batch], AUDIO_PADDING_TOKEN)
    pad_text = tokens_padding([item[1] for item in batch], TEXT_PADDING_TOKEN)
    return (pad_wav, pad_text)


def main():
    parser = argparse.ArgumentParser(
        prog='WavTokens Audio Adapter training script')
    parser.add_argument('--config', required=True, type=str)
    args = parser.parse_args()

    config = read_config(args.config)

    print('Loading dataset')
    train_dataset = load_from_disk(config.dataset.librispeech_train)
    # train_dataset = LimitedDataset(train_dataset.select(list(range(50_000))))
    train_dataset = LimitedDataset(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=config.train.dl_workers
    )

    print('Loading model')
    tokenizer = load_qwen_tokenizer()
    qwen_emb = load_qwen_emb()
    if hasattr(config.model, 'load_checkpoint'):
        adapter = WavTokensEncoder.load(
            config.model.load_checkpoint, AUDIO_TOKENS, QWEN_EMB_SIZE, config.model.num_layers)
    else:
        adapter = WavTokensEncoder(
            AUDIO_TOKENS, QWEN_EMB_SIZE, config.model.num_layers, add_rope=True)
    adapter = adapter.to(device)
    adapter.train()
    opt = torch.optim.Adam(adapter.parameters(), config.opt.warmup_lr)

    if config.wandb.enabled:
        print('Initialising W&B')
        wandb.login(key=WANDB_TOKEN)
        wandb.init(
            project=config.wandb.project,
            config=config,
            name=config.wandb.run_name,
            notes=config.wandb.notes,
            # disable system logging
            settings=wandb.Settings(_disable_stats=True, _disable_meta=True)
        )
    else:
        print('Warning: wandb is not enabled')

    warmup = True
    batch_step = 0
    global_step = 0
    best_loss = None
    loss = 0

    print('Start training')
    opt.zero_grad()
    for epoch in range(config.train.epoches):
        print(f'Start epoch {epoch}')

        adapter.train()
        for batch in tqdm(train_dataloader):
            audio_mask = (batch[0] != AUDIO_PADDING_TOKEN)
            text_emb = get_qwen_emb(qwen_emb, batch[1]).to(device)
            text_mask = (batch[1] != TEXT_PADDING_TOKEN)
            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                wav_emb = adapter(batch[0].to(device))
                colbert_loss, matrix_score = colBertMatrixBlockLoss(
                    wav_emb,
                    text_emb,
                    audio_mask,
                    text_mask
                )
            batch_loss = 1 / config.train.gradient_accumulation_steps * colbert_loss
            batch_loss.backward()
            batch_step += 1
            loss += batch_loss.item()

            if batch_step == config.train.gradient_accumulation_steps:
                grad_norm = torch.sqrt(
                    sum([torch.norm(param.grad)**2 for param in adapter.parameters()]))
                torch.nn.utils.clip_grad_norm_(
                    adapter.parameters(), config.train.grad_clip)
                opt.step()
                opt.zero_grad()

                print(
                    f'step {global_step}; loss: {loss}; grad_norm: {grad_norm}')
                if config.wandb.enabled:
                    report_loss(global_step, loss, grad_norm)

                batch_step = 0
                global_step += 1
                loss = 0

                if global_step % config.train.save_freq == 0:
                    adapter.save(config.train.save_checkpoint_path)
                    if best_loss is None or loss < best_loss:
                        best_loss = loss
                        adapter.save(
                            config.train.save_best_checkpoint_path)

                if global_step % config.log.sample_freq == 0:
                    text = [tokenizer.decode(item, skip_special_tokens=True)
                            for item in batch[1]]
                    if config.wandb.enabled:
                        report_sample(text, matrix_score,
                                      config.log.num_samples)

                if warmup and global_step == config.opt.warmup_steps:
                    warmup = False
                    for group in opt.param_groups:
                        group['lr'] = config.opt.lr

            torch.cuda.empty_cache()
            gc.collect()


if __name__ == '__main__':
    main()
