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


def normalize_text(utterances: Collection[str]) -> List[str]:
    global whisper_tokenizer
    return list(map(lambda s: whisper_tokenizer.normalize(s), utterances))


def calculate_wer(input: Collection[str], target: Collection[str]) -> float:
    return jiwer.wer(normalize_text(input), normalize_text(target))


def calculate_wer_single(input: str, target: str) -> float:
    return calculate_wer([input], [target])


def measure_model_wer(model: ASRBaseModel, loader: DataLoader) -> float:
    expected = []
    actual = []
    for batch in loader:
        expected.extend(batch['text'])
        actual.extend(model.generate(batch['audio']))
    wer = jiwer.wer(normalize_text(expected), normalize_text(actual))
    return wer
