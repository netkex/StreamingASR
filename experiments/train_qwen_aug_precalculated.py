import argparse
import wandb
import os

import numpy as np

import torch
import torch.nn as nn

from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, EvalPrediction
from transformers.integrations import WandbCallback
from datasets import load_from_disk

from StreamASRLib.eval import calculate_wer
from StreamASRLib.model import CustomWavTokenizer, WavQwenModel


WANDB_TOKEN = os.environ['WANDB_TOKEN']
# os.environ['WANDB_PROJECT'] = 'Qwen05b-hf-full-train'
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set torch seed
torch.manual_seed(13)


def build_ref_seq(tokenizer, output: torch.Tensor) -> str:
    tokens = torch.argmax(output, dim=-1)
    return tokenizer.decode(tokens.cpu())


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
            try:
                self.wer_hist.append(calculate_wer([ref_seq], [orig_seq]))
            except Exception as e:
                print(
                    f'[WARNING] Failed to calulate WER* because of error "{e}"')
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
                torch.tensor(sample['input_ids'][audio_start:audio_end]) - WavQwenModel.AUDIO_TOKEN_THRESHOLD).numpy()
            target_ = self.tokenizer.decode(
                torch.tensor(sample['input_ids'][text_start:text_end]))
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
        sample_ds = [ds[ind.item()] for ind in sample_ind]
        return sample_ds

    def log_val_table(self):
        sample_ds = self._get_sample_ds(self.val_dataset)
        sample_table = self._build_sample_table(sample_ds)
        self._wandb.log({'Validation samples': sample_table})

    def log_train_table(self):
        sample_ds = self._get_sample_ds(self.train_dataset)
        sample_table = self._build_sample_table(sample_ds)
        self._wandb.log({'Train samples': sample_table})

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        self.log_val_table()
        self.log_train_table()


def main():
    parser = argparse.ArgumentParser(prog='Qwen training script')
    parser.add_argument(
        '--train-path', default='/home/netkex/datasets/custom/librispeech-train-75tks-aug-final', type=str)
    parser.add_argument(
        '--test-path',  default='/home/netkex/datasets/custom/librispeech-test-75tks', type=str)
    parser.add_argument(
        '--val-length', default=512, type=int)
    parser.add_argument(
        '--wav-tokenizer-config', default='/home/netkex/models/WavTokenizer/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml', type=str)
    parser.add_argument(
        '--wav-tokenizer-model', default='/home/netkex/models/WavTokenizer/wavtokenizer_large_speech_320_24k.ckpt', type=str)
    parser.add_argument(
        '--run-name', default='iteration-3.0', type=str)
    parser.add_argument(
        '--output-dir', default='/home/netkex/outputs/qwen05-train-75-tks-aug-v2', type=str)
    parser.add_argument(
        '--max-steps', default=20_000, type=int)
    parser.add_argument(
        '--checkpoint', default=None, type=str)
    parser.add_argument(
        '--notes', default=None, type=str)
    args = parser.parse_args()

    # Prepare datasets
    print("Loading datasets")

    # Load data
    train_ds = load_from_disk(args.train_path)
    test_ds = load_from_disk(args.test_path)
    val_ds = test_ds.select(range(args.val_length))

    # Load model
    print('Loading model')
    if args.checkpoint is not None:
        qwen_model = WavQwenModel.load_checkpoint(args.checkpoint)
    else:
        qwen_model = WavQwenModel.init()
    # does not work with multiple GPU
    # qwen_model.gradient_checkpointing_enable(
    #     gradient_checkpointing_kwargs={"use_reentrant": False})
    qwen_model.model.train()

    wav_tokenizer = CustomWavTokenizer(
        args.wav_tokenizer_config, args.wav_tokenizer_model, device='cpu')

    # Init wandb
    wandb.login(key=WANDB_TOKEN)
    wandb.init(
        project="Qwen05b-wav-75ts-aug-train",
        config={},
        name=args.run_name,
        notes=args.notes,
        # disable system logging
        # settings=wandb.Settings(_disable_stats=True, _disable_meta=True)
    )

    # Train model
    wer_metric = WerMetric(qwen_model.tokenizer)

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy='steps', eval_steps=25, batch_eval_metrics=True, include_inputs_for_metrics=True,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        dataloader_num_workers=12, dataloader_prefetch_factor=6,
        gradient_accumulation_steps=32,
        learning_rate=2e-4, max_grad_norm=25.0,
        max_steps=args.max_steps,
        report_to='wandb', logging_strategy='steps', logging_steps=1,
        save_strategy='steps', save_steps=50, save_total_limit=3, load_best_model_at_end=True,
        label_names=['label_text_start', 'label_text_end'],
        remove_unused_columns=False,
        # gradient_checkpointing=True
    )

    trainer = CustomTrainer(
        model=qwen_model.model,
        args=train_args,
        data_collator=DataCollatorWithPadding(qwen_model.tokenizer),
        train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=wer_metric,
    )

    wandb_callback = WandbProgressCallback(
        trainer,
        qwen_model.tokenizer,
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
