import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Dict

import torch
import torchaudio
from datasets import Dataset


# https://github.com/saveriyo/WavTokenizer
from wavtokenizer.encoder.utils import convert_audio
from wavtokenizer.decoder.pretrained import WavTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DatasetWrapper:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        audio, sample_rate, text, _, _, _ = self.dataset[item]
        return {
            'audio': audio,
            'text': text,
            'sr': sample_rate
        }


def prepare_dir(dir: str):
    Path(dir).mkdir(parents=True, exist_ok=True)


def download_dataset(dir: str, split: str):
    dataset = torchaudio.datasets.LIBRISPEECH(
        root=dir,
        url=split,
        download=True
    )
    return DatasetWrapper(dataset)


def download_wavtokenizer(model_path: str, config_path: str):
    global device
    return WavTokenizer.from_pretrained0802(config_path, model_path).to(device)


def extract_wav_tokens(wav_model, audio_item: Dict[str, int | torch.Tensor | str]) -> torch.Tensor:
    global device
    resampled_audio = convert_audio(
        audio_item['audio'], audio_item['sr'], 24000, 1)
    bandwidth_id = torch.tensor([0])
    _, discrete_code = wav_model.encode_infer(
        resampled_audio.to(device), bandwidth_id=bandwidth_id)
    return discrete_code


def generate(wav_model, dataset: DatasetWrapper):
    for audio_item in tqdm(dataset):
        wav_tokens = extract_wav_tokens(wav_model, audio_item)
        yield {'wav-tokens': wav_tokens.flatten(), 'text': audio_item['text']}


def main():
    parser = argparse.ArgumentParser(prog='Raw Wav Tokenizer Tool')
    parser.add_argument('-d', '--download-path', required=True, type=str)
    parser.add_argument('--split', required=True, type=str)
    parser.add_argument('-c', '--config-path', required=True, type=str)
    parser.add_argument('-m', '--model-path', required=True, type=str)
    parser.add_argument('-s', '--save-path', required=True, type=str)
    args = parser.parse_args()

    prepare_dir(args.download_path)
    prepare_dir(args.save_path)

    dataset = download_dataset(args.download_path, args.split)
    wav_model = download_wavtokenizer(args.model_path, args.config_path)
    raw_wav_ds = Dataset.from_generator(
        generate, gen_kwargs={'wav_model': wav_model, 'dataset': dataset}
    )
    raw_wav_ds.save_to_disk(args.save_path)


if __name__ == "__main__":
    main()
