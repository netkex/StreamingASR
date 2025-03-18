import numpy as np
import torch
import wandb
from transformers import AutoTokenizer, Trainer
from transformers.integrations import WandbCallback

from StreamASRLib.model import CustomWavTokenizer, WavQwenBaseModel


class WandbAudioSampleCallback(WandbCallback):
    def __init__(self,
                 trainer: Trainer,
                 tokenizer: AutoTokenizer,
                 wav_tokenizer: CustomWavTokenizer,
                 train_dataset,
                 val_dataset,
                 num_samples: int = 10,
                 audio_token_threshold: int = WavQwenBaseModel.AUDIO_TOKEN_THRESHOLD):
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.wav_tokenizer = wav_tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.num_samples = num_samples
        self.audio_token_threshold = audio_token_threshold

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
                torch.tensor(sample['input_ids'][audio_start:audio_end]) - self.audio_token_threshold).numpy()
            target_ = self.tokenizer.decode(
                torch.tensor(sample['input_ids'][text_start:text_end]))
            predicted.append(predicted_)
            target.append(target_)
            audio.append(audio_)

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
