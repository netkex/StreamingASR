import wandb
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, EvalPrediction
from transformers.integrations import WandbCallback
from datasets import load_dataset, load_from_disk, concatenate_datasets

from tqdm.notebook import tqdm
from typing import Dict, List

from StreamASRLib.eval import calculate_wer


device = 'cuda' if torch.cuda.is_available() else 'cpu'

WANDB_TOKEN = os.environ['WANDB_TOKEN']
os.environ['WANDB_PROJECT'] = 'Qwen05b-hf-full-train'
# os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Set config
QWEN_REPO = 'Qwen/Qwen2.5-0.5B'

EOS_TOKEN = 151643
PADDING_TOKEN = EOS_TOKEN
AUDIO_TOKENS = 4096
AUDIO_TOKEN_THRESHOLD = 151936
BOA_TOKEN = AUDIO_TOKEN_THRESHOLD + AUDIO_TOKENS
EOA_TOKEN = BOA_TOKEN + 1
EXT_TOKENS = AUDIO_TOKENS + 2

DS_TRAIN_360_RAW = '/storage/netkex/datasets/custom/librispeech-train-clean-360-raw'
DS_TRAIN_500_RAW = '/storage/netkex/datasets/custom/librispeech-train-other-500-raw'
DS_TEST_RAW = '/storage/netkex/datasets/custom/librispeech-test-clean-raw'

DS_TRAIN = '/storage/netkex/datasets/custom/librispeech-train-2'
DS_TEST = '/storage/netkex/datasets/custom/librispeech-test-2'

tokenizer = None


def prepare_ds(ds: Dataset):
    ds_ = ds.remove_columns(['wav-tokens', 'text'])
    return ds_


def build_ref_seq(output: torch.Tensor) -> str:
    global tokenizer
    tokens = torch.argmax(output, dim=-1)
    return tokenizer.decode(tokens.cpu())

# Custom embedding class for new wav-tokens
class ExtendedEmbedding(nn.Module):
    def __init__(self, base_emb: nn.Embedding, ext_tokens_size: int, ext_token_threshold: int):
        super().__init__()
        self.base_emb = base_emb
        self.hidden_size = base_emb.embedding_dim
        self.ext_emb = nn.Embedding(ext_tokens_size, self.hidden_size)
        self.ext_token_threshold = ext_token_threshold

    def freeze_base_emb(self):
        for param in self.base_emb.parameters():
            param.requires_grad = False

    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        ext_token_mask = (input_tokens >= self.ext_token_threshold)
        inp_emb = torch.zeros(list(
            input_tokens.shape) + [self.hidden_size], requires_grad=True).to(self.base_emb.weight.device)
        inp_emb[~ext_token_mask] = self.base_emb(input_tokens[~ext_token_mask])
        inp_emb[ext_token_mask] = self.ext_emb(
            input_tokens[ext_token_mask] - self.ext_token_threshold)
        return inp_emb


# Custom trainer class with cross-entropy loss only for text tokens
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = nn.CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False):
        label_text_start = inputs.label_text_start
        label_text_end = inputs.label_text_end
        output = model(input_ids=inputs.input_ids,
                       attention_mask=inputs.attention_mask)

        logits = []
        targets = []
        for id, (start, end) in enumerate(zip(label_text_start, label_text_end)):
            logits.append(output.logits[id, (start - 1):(end - 1), :])
            targets.append(inputs.input_ids[id, start:end])

        logits_ = torch.concatenate(logits, dim=0)
        targets_ = torch.concatenate(targets, dim=0)
        loss = self.loss(logits_, targets_)
        return (loss, output) if (return_outputs) else loss


# Metrics and logging stuff
class WerMetric:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.wer_hist = []

    def __call__(self, eval_pred: EvalPrediction, compute_result: bool) -> float:
        logits = eval_pred.predictions
        input_ids = getattr(eval_pred, 'inputs')['input_ids']
        label_text_start = eval_pred.label_ids[0]
        label_text_end = eval_pred.label_ids[1]

        for id, (start, end) in enumerate(zip(label_text_start, label_text_end)):
            ref_seq = build_ref_seq(logits[id, (start - 1):(end - 2), :])
            orig_seq = self.tokenizer.decode(
                input_ids[id, start:(end - 1)].detach().cpu())
            self.wer_hist.append(calculate_wer([ref_seq], [orig_seq]))
        if compute_result:
            wer_ = np.mean(self.wer_hist)
            self.wer_hist = []
            return {'WER*': wer_}


class WandbProgressCallback(WandbCallback):
    def __init__(self, trainer, tokenizer, val_dataset, num_samples=10):
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.val_dataset = val_dataset
        self.num_samples = num_samples

    def log_sample_table(self):
        sample_dataset = self.val_dataset.select(
            np.random.choice(len(self.val_dataset), size=self.num_samples))
        output = self.trainer.predict(sample_dataset)

        refs = []
        targets = []
        for i, sample in enumerate(sample_dataset):
            start, end = sample['label_text_start'], sample['label_text_end']
            ref = self.tokenizer.decode(
                np.argmax(output.predictions[i, (start - 1):(end - 1), :], axis=-1))
            target = self.tokenizer.decode(sample['input_ids'][start:end])
            refs.append(ref)
            targets.append(target)

        columns = ['Reference', 'Target']
        data = [[ref, target] for ref, target in zip(refs, targets)]
        samples_table = self._wandb.Table(data=data, columns=columns)
        self._wandb.log({'Sample predictions': samples_table})

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        self.log_sample_table()


def main():
    global tokenizer
    
    # Load data
    ds_train = load_from_disk(DS_TRAIN)
    ds_test = load_from_disk(DS_TEST)

    ds_train_ = prepare_ds(ds_train)
    ds_test_ = prepare_ds(ds_test)
    ds_val_ = ds_test_.select(range(250))

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, use_fast=True)
    qwen_model = AutoModelForCausalLM.from_pretrained(QWEN_REPO)

    base_emb = qwen_model.model.embed_tokens
    ext_emb = ExtendedEmbedding(base_emb, EXT_TOKENS, AUDIO_TOKEN_THRESHOLD)
    # ext_emb.freeze_base_emb()
    qwen_model.model.embed_tokens = ext_emb

    qwen_model.train()
    # does not work with multiple GPU
    # qwen_model.gradient_checkpointing_enable(
    #     gradient_checkpointing_kwargs={"use_reentrant": False})

    # Init wandb
    wandb.login(key=WANDB_TOKEN)
    wandb.init(
        project="Qwen05b-hf-full-train",
        config={},
        name="iteration-2.1",
        # disable system logging
        settings=wandb.Settings(_disable_stats=True, _disable_meta=True)
    )

    # Train model
    wer_metric = WerMetric(tokenizer)

    train_args = TrainingArguments(
        output_dir='/storage/netkex/outputs/qwen05-train',
        eval_strategy='steps', eval_steps=25, batch_eval_metrics=True, include_inputs_for_metrics=True,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        dataloader_num_workers=8, dataloader_prefetch_factor=4,
        gradient_accumulation_steps=32,
        learning_rate=6e-5, max_grad_norm=25.0,
        max_steps=10_000,
        report_to='wandb', logging_strategy='steps', logging_steps=1,
        save_strategy='steps', save_steps=50, save_total_limit=3, load_best_model_at_end=True,
        label_names=['label_text_start', 'label_text_end'],
        remove_unused_columns=False,
        # gradient_checkpointing=True
    )

    trainer = CustomTrainer(
        model=qwen_model,
        args=train_args,
        data_collator=DataCollatorWithPadding(tokenizer),
        train_dataset=ds_train_, eval_dataset=ds_val_,
        compute_metrics=wer_metric
    )

    wandb_callback = WandbProgressCallback(trainer, tokenizer, ds_val_)
    trainer.add_callback(wandb_callback)

    try:
        trainer.train()
        trainer.save_model()
    finally:
        wandb.finish()


if __name__ == '__main__':
    main()
