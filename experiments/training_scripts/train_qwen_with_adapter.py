import argparse
import os

import torch
import torchaudio
import wandb
from datasets import concatenate_datasets, load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (AutoTokenizer, DataCollatorWithPadding,
                          TrainingArguments)

from StreamASRLib.dataset import (LibriDataset, MergeDataset, SpecAugmentation,
                                  SpecAugmentationConfig, WavTextDataset)
from StreamASRLib.logging import WandbAudioSampleCallback
from StreamASRLib.model import (CustomWavTokenizer, WavQwenModelWithEncoder,
                                WavTokensEncoder)
from StreamASRLib.training import WavAudioTrainer, WerMetric, read_config

WANDB_TOKEN = os.environ['WANDB_TOKEN']

os.environ['WANDB_PROJECT'] = 'Qwen05b-hf-full-train'
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set torch seed
torch.manual_seed(42)


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

        input_ids = torch.tensor(
            [WavQwenModelWithEncoder.BOA_TOKEN] +
            [token + WavQwenModelWithEncoder.AUDIO_TOKEN_THRESHOLD for token in wav_tokens] +
            [WavQwenModelWithEncoder.EOA_TOKEN] +
            text_tokens +
            [WavQwenModelWithEncoder.EOS_TOKEN]
        )
        attention_mask = torch.ones_like(input_ids)
        label_text_start = 2 + len(wav_tokens)
        label_text_end = input_ids.shape[0]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label_text_start': label_text_start,
            'label_text_end': label_text_end
        }


def main():
    parser = argparse.ArgumentParser(prog='Qwen with encoder training script')
    parser.add_argument('--config', required=True, type=str)
    args = parser.parse_args()

    config = read_config(args.config)

    # Prepare datasets
    print("Loading datasets")
    train_dataset = load_from_disk(config.dataset.librispeech_train)
    # train_dataset = LimitedDataset(train_dataset.select(list(range(10_000))))
    train_dataset = LimitedDataset(train_dataset)

    val_dataset = load_from_disk(config.dataset.librispeech_test)
    val_dataset = LimitedDataset(val_dataset.select(
        torch.randint(len(val_dataset), size=(config.dataset.val_size,)).tolist()))

    wav_tokenizer = CustomWavTokenizer(
        config.wav_tokenizer.config, config.wav_tokenizer.model, device='cpu')

    # Load model
    print('Loading model')
    if hasattr(config.model, 'checkpoint'):
        raise NotImplementedError('Not supported yet')
    else:
        adapter = WavTokensEncoder(
            # TODO: replace +2 with something normal
            WavQwenModelWithEncoder.AUDIO_TOKENS + 2,
            896,  # TODO: make constant somewhere
            config.model.encoder.num_layers,
            add_rope=True
        )
        adapter.load_state_dict(torch.load(
            config.model.encoder.checkpoint, weights_only=True))
        qwen_model = WavQwenModelWithEncoder.init_with_encoder(adapter)
        if config.model.freeze_emb:
            qwen_model.freeze_emb()
    qwen_model.model.train()

    # does not work with multiple GPU
    # qwen_model.gradient_checkpointing_enable(
    #     gradient_checkpointing_kwargs={"use_reentrant": False})

    # Init wandb
    wandb.login(key=WANDB_TOKEN)
    wandb.init(
        project=config.wandb.project,
        config=config,
        name=config.wandb.run_name,
        notes=config.wandb.notes,
        # disable system logging
        # settings=wandb.Settings(_disable_stats=True, _disable_meta=True)
    )

    # Train model
    wer_metric = WerMetric(qwen_model.tokenizer)

    train_args = TrainingArguments(
        output_dir=config.training.output_dir,
        eval_strategy='steps',
        eval_steps=config.training.eval_steps,
        batch_eval_metrics=True,
        include_inputs_for_metrics=True,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        bf16=config.training.bf16,
        bf16_full_eval=config.training.bf16_full_eval,
        dataloader_num_workers=config.training.dataloader_num_workers,
        dataloader_prefetch_factor=config.training.dataloader_prefetch_factor,
        learning_rate=config.training.learning_rate,
        lr_scheduler_type=config.training.lr_scheduler_type,
        warmup_steps=config.training.warmup_steps,
        max_grad_norm=config.training.max_grad_norm,
        max_steps=config.training.max_steps,
        report_to='wandb', logging_strategy='steps', logging_steps=1,
        save_strategy='steps', save_steps=config.training.save_steps, save_total_limit=3, load_best_model_at_end=True,
        label_names=['label_text_start', 'label_text_end'],
        remove_unused_columns=False,
        # gradient_checkpointing=True
    )

    trainer = WavAudioTrainer(
        model=qwen_model.model,
        args=train_args,
        data_collator=DataCollatorWithPadding(qwen_model.tokenizer),
        train_dataset=train_dataset, eval_dataset=val_dataset,
        compute_metrics=wer_metric,
    )

    wandb_callback = WandbAudioSampleCallback(
        trainer,
        qwen_model.tokenizer,
        wav_tokenizer,
        train_dataset,
        val_dataset
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
