import torch
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import Dataset

from typing import Dict


class WavTokenTrainDataset(Dataset):
    def __init__(self,
                 df: pd.core.frame.DataFrame,
                 tokenizer: AutoTokenizer,
                 boa_token: int,
                 eoa_token: int,
                 pre_audio_prompt: str | None = None,
                 post_audio_prompt: str | None = None,
                 bos_token: int | None = None,
                 eos_token: int | None = None):
        self.df = df
        self.tokenizer = tokenizer

        self.pre_audio_prompt = pre_audio_prompt
        self.pre_audio_prompt_tokens = WavTokenTrainDataset.tokenize(
            tokenizer, pre_audio_prompt)

        self.post_audio_prompt = post_audio_prompt
        self.post_audio_prompt_tokens = WavTokenTrainDataset.tokenize(
            tokenizer, post_audio_prompt)

        self.bos_token = bos_token
        self.eos_token = eos_token
        self.bos_tensor = torch.tensor([bos_token]) if (
            bos_token is not None) else None
        self.eos_tensor = torch.tensor([eos_token]) if (
            bos_token is not None) else None

        self.boa_token = boa_token
        self.eoa_token = eoa_token
        self.boa_tensor = torch.tensor([boa_token])
        self.eoa_tensor = torch.tensor([eoa_token])

    @staticmethod
    def tokenize(tokenizer: AutoTokenizer, uttence: str | None) -> torch.Tensor | None:
        if uttence is None:
            return None
        return tokenizer(uttence, add_special_tokens=False, return_tensors='pt').input_ids.flatten()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item: int) -> Dict[str, str | torch.Tensor]:
        df_row = self.df.iloc[item]
        wav_tokens = torch.tensor(
            list(map(int, df_row['wav-tokens'][1:-1].split(', '))))

        text = df_row['text']
        text_tokens = self.tokenizer(
            text.lower(), add_special_tokens=False, return_tensors='pt').input_ids.flatten()

        text_len = len(text_tokens)
        sequence_parts = []
        if self.bos_tensor is not None:
            sequence_parts.append(self.bos_tensor)
        if self.pre_audio_prompt_tokens is not None:
            sequence_parts.append(self.pre_audio_prompt_tokens)
        sequence_parts.extend([self.boa_tensor, wav_tokens, self.eoa_tensor])
        if self.post_audio_prompt_tokens is not None:
            sequence_parts.append(self.post_audio_prompt_tokens)
        sequence_parts.append(text_tokens)
        if self.eos_tensor is not None:
            sequence_parts.append(self.eos_tensor)
            text_len += 1

        sequence = torch.concat(sequence_parts)
        text_end = len(sequence)
        text_start = text_end - text_len
        return {
            'sequence': sequence,
            'text': text,
            'text-start': text_start,
            'text-end': text_end
        }
