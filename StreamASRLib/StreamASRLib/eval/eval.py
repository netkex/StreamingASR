from dataclasses import dataclass
from typing import Collection, List

import jiwer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import WhisperTokenizer

from StreamASRLib.model import ASRBaseModel

whisper_tokenizer = WhisperTokenizer.from_pretrained(
    "openai/whisper-large-v2", language="english")


@dataclass
class EvalMetric:
    wer: float


def normalize(utterances: Collection[str]) -> List[str]:
    global whisper_tokenizer
    return list(map(lambda s: whisper_tokenizer.normalize(s), utterances))


def calculate_wer(model: ASRBaseModel, loader: DataLoader) -> float:
    expected = []
    actual = []
    for batch in loader:
        expected.extend(batch['text'])
        actual.extend(model.generate(batch['audio']))
    wer = jiwer.wer(normalize(expected), normalize(actual))
    return wer
