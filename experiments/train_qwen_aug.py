import argparse
import wandb
import os
import multiprocessing

import numpy as np

import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, EvalPrediction
from transformers.integrations import WandbCallback
from datasets import load_dataset, load_from_disk, concatenate_datasets

from StreamASRLib.eval import calculate_wer
from StreamASRLib.dataset import MergeDataset, WavTextDataset, LibriDataset, SpecAugmentation, SpecAugmentationConfig
from StreamASRLib.model import CustomWavTokenizer


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

# Set torch seed
torch.manual_seed(42)


def librispeech_cut_long(ds, str_length: str):
    ind = []
    for i in range(len(ds)):
        meta = ds.get_metadata(i)
        if len(meta[2]) <= str_length:
            ind.append(i)
    return torch.utils.data.Subset(ds, ind)


def load_train_ds(
    libri_train_clean_path: str,
    libri_train_other_path: str,
    wav_tokenizer: CustomWavTokenizer,
    tokenizer: AutoTokenizer,
    augmentation: SpecAugmentation,
    length_cut: int = 300,
    drop_tokens: bool = False,
    drop_tokens_p: float = 0.0,
):
    train_clean_ds = torchaudio.datasets.LIBRISPEECH(
        root=libri_train_clean_path,
        url='train-clean-360',
        download=False
    )
    train_other_ds = torchaudio.datasets.LIBRISPEECH(
        root=libri_train_other_path,
        url='train-other-500',
        download=False
    )
    train_clean_cut_ds = librispeech_cut_long(train_clean_ds, length_cut)
    train_other_cut_ds = librispeech_cut_long(train_other_ds, length_cut)
    train_ds = MergeDataset(
        [
            LibriDataset(train_clean_cut_ds),
            LibriDataset(train_other_cut_ds)
        ]
    )
    wav_train_ds = WavTextDataset(
        wav_model=wav_tokenizer,
        tokenizer=tokenizer,
        librispeech_dataset=train_ds,
        spec_augmentation=augmentation,
        eos_token=EOS_TOKEN,
        boa_token=BOA_TOKEN,
        eoa_token=EOA_TOKEN,
        audio_token_threshold=AUDIO_TOKEN_THRESHOLD,
        drop_tokens=drop_tokens,
        drop_token_prob=drop_tokens_p,
        add_text=False
    )
    return wav_train_ds


def load_val_ds(
    libri_test_clean_path: str,
    wav_tokenizer: CustomWavTokenizer,
    tokenizer: AutoTokenizer,
    val_length: int = 250,
    length_cut: int = 300,
):
    test_clean_ds = torchaudio.datasets.LIBRISPEECH(
        root=libri_test_clean_path,
        url='test-clean',
        download=False
    )
    test_clean_ds = librispeech_cut_long(test_clean_ds, length_cut)
    val_ind = torch.randint(len(test_clean_ds), size=(val_length,))
    val_clean_ds = torch.utils.data.Subset(test_clean_ds, val_ind)
    val_clean_ds = LibriDataset(val_clean_ds)
    wav_test_ds = WavTextDataset(
        wav_model=wav_tokenizer,
        tokenizer=tokenizer,
        librispeech_dataset=val_clean_ds,
        spec_augmentation=lambda wav: wav,
        eos_token=EOS_TOKEN,
        boa_token=BOA_TOKEN,
        eoa_token=EOA_TOKEN,
        audio_token_threshold=AUDIO_TOKEN_THRESHOLD,
        drop_tokens=False,
        drop_token_prob=0.0,
        add_text=False
    )
    return wav_test_ds


def build_ref_seq(tokenizer, output: torch.Tensor) -> str:
    tokens = torch.argmax(output, dim=-1)
    return tokenizer.decode(tokens.cpu())


class ExtendedEmbedding(nn.Module):
    '''
    Custom embedding class for wav-tokens
    '''

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


class CustomTrainer(Trainer):
    '''
    Custom trainer class with cross-entropy loss only for text tokens
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = nn.CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False):
        label_text_start = inputs.label_text_start
        label_text_end = inputs.label_text_end
        output = model(input_ids=inputs.input_ids,
                       attention_mask=inputs.attention_mask)

        logits = output.logits.float()
        batch_size = logits.shape[0]
        loss = 0

        for id, (start, end) in enumerate(zip(label_text_start, label_text_end)):
            seq_logits = logits[id, (start - 1):(end - 1), :]
            seq_target = inputs.input_ids[id, start:end]
            loss += self.loss(seq_logits, seq_target)
        loss /= batch_size

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
            ref_seq = build_ref_seq(
                self.tokenizer, logits[id, (start - 1):(end - 2), :])
            orig_seq = self.tokenizer.decode(
                input_ids[id, start:(end - 1)].detach().cpu())
            self.wer_hist.append(calculate_wer([ref_seq], [orig_seq]))
        if compute_result:
            wer_ = np.mean(self.wer_hist)
            self.wer_hist = []
            return {'WER*': wer_}


class WandbProgressCallback(WandbCallback):
    def __init__(self,
                 trainer: Trainer,
                 tokenizer,
                 wav_tokenizer: CustomWavTokenizer,
                 train_dataset,
                 val_dataset,
                 num_samples=10):
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.wav_tokenizer = wav_tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.num_samples = num_samples

    def _build_sample_table(self, sample_dataset):
        output = self.trainer.predict(sample_dataset)
        target = []
        predicted = []
        audio = []

        for i, sample in enumerate(sample_dataset):
            text_start, text_end = sample['label_text_start'], sample['label_text_end']
            audio_start, audio_end = 1, text_start - 1
            predicted_ = self.tokenizer.decode(
                np.argmax(output.predictions[i, (text_start - 1):(text_end - 1), :], axis=-1))
            audio_ = self.wav_tokenizer.decode(
                sample['input_ids'][audio_start:audio_end] - AUDIO_TOKEN_THRESHOLD).numpy()
            target_ = self.tokenizer.decode(
                sample['input_ids'][text_start:text_end])
            predicted.append(predicted_)
            audio.append(audio_)
            target.append(target_)

        columns = ['Audio', 'Target', 'Predicted*']
        data = [[wandb.Audio(audio, self.wav_tokenizer.WAV_SR), target, predicted]
                for audio, target, predicted in zip(audio, target, predicted)]
        sample_table = self._wandb.Table(data=data, columns=columns)
        return sample_table

    def _get_sample_ds(self, ds):
        sample_ind = torch.randint(len(ds), size=(self.num_samples,))
        sample_ds = [ds[ind] for ind in sample_ind]
        return sample_ds

    def log_val_table(self):
        sample_ds = self._get_sample_ds(self.val_dataset)
        sample_table = self._build_sample_table(sample_ds)
        self._wandb.log({'Validation samples': sample_table})

    def log_train_table(self):
        sample_ds = self._get_sample_ds(self.val_dataset)
        sample_table = self._build_sample_table(sample_ds)
        self._wandb.log({'Train samples': sample_table})

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        self.log_val_table()
        self.log_train_table()


def main():
    parser = argparse.ArgumentParser(prog='Qwen training script')
    parser.add_argument(
        '--librispeech-train-clean', default='/home/netkex/datasets/original/librispeech-train-clean-360', type=str)
    parser.add_argument(
        '--librispeech-train-other', default='/home/netkex/datasets/original/librispeech-train-other-500', type=str)
    parser.add_argument(
        '--librispeech-test-clean', default='/home/netkex/datasets/original/librispeech-test-clean', type=str)
    parser.add_argument(
        '--wav-tokenizer-config', default='/home/netkex/models/WavTokenizer/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml', type=str)
    parser.add_argument(
        '--wav-tokenizer-model', default='/home/netkex/models/WavTokenizer//wavtokenizer_large_unify_600_24k.ckpt', type=str)
    parser.add_argument(
        '--val-length', default=250, type=int)
    parser.add_argument(
        '--cut-length', default=300, type=int)
    parser.add_argument(
        '--run-name', default='iteration-0', type=str)
    parser.add_argument(
        '--output-dir', default='/home/netkex/outputs/qwen05-train-40-tks-aug', type=str)
    parser.add_argument(
        '--max-steps', default=20_000, type=int)
    args = parser.parse_args()

    # Prepare datasets
    print("Loading datasets")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, use_fast=True)
    wav_tokenizer = CustomWavTokenizer(
        args.wav_tokenizer_config, args.wav_tokenizer_model, device='cpu')
    augmentation_config = SpecAugmentationConfig(
        apply_aug_p=0.75,
        apply_speed=True,
        speed_p=0.5,
        apply_masking=True,
        masking_p=0.5,
        freq_mask_param=150,
        apply_noise=True,
        noise_p=0.5,
        noise_snr=32
    )
    augmentation = SpecAugmentation(n_fft=1024, config=augmentation_config)

    # Load data
    train_ds = load_train_ds(
        args.librispeech_train_clean,
        args.librispeech_train_other,
        wav_tokenizer,
        tokenizer,
        augmentation,
        args.cut_length)
    val_ds = load_val_ds(
        args.librispeech_test_clean,
        wav_tokenizer,
        tokenizer,
        args.val_length,
        args.cut_length
    )

    # Load model
    print('Loading model')
    qwen_model = AutoModelForCausalLM.from_pretrained(QWEN_REPO)
    base_emb = qwen_model.model.embed_tokens
    ext_emb = ExtendedEmbedding(base_emb, EXT_TOKENS, AUDIO_TOKEN_THRESHOLD)
    # ext_emb.freeze_base_emb()
    qwen_model.model.embed_tokens = ext_emb

    qwen_model.train()

    # does not work with multiple GPU
    # qwen_model.gradient_checkpointing_enable(
    #     gradient_checkpointing_kwargs={"use_reentrant": False})

#     # Init wandb
    wandb.login(key=WANDB_TOKEN)
    wandb.init(
        project="Qwen05b-wav-40ts-aug-train",
        config={},
        name=args.run_name,
        # disable system logging
        settings=wandb.Settings(_disable_stats=True, _disable_meta=True)
    )

    # Train model
    wer_metric = WerMetric(tokenizer)

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy='steps', eval_steps=25, batch_eval_metrics=True, include_inputs_for_metrics=True,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        dataloader_num_workers=8, dataloader_prefetch_factor=4,
        gradient_accumulation_steps=32,
        learning_rate=6e-5, max_grad_norm=25.0,
        max_steps=args.max_steps,
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
        train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=wer_metric,
    )

    wandb_callback = WandbProgressCallback(
        trainer,
        tokenizer,
        wav_tokenizer,
        train_ds,
        val_ds
    )
    trainer.add_callback(wandb_callback)

    try:
        print('Start training')
        trainer.train()
        trainer.save_model()
    finally:
        print('Finished training')
        wandb.finish()


if __name__ == '__main__':
    main()
