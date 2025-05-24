import torch
from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram

from .utils import LibriDataset


class LibriLogMel(Dataset):
    def __init__(self,
                 libri_ds: LibriDataset,
                 n_mels: int = 80,
                 sample_rate: int = 16_000,
                 n_fft: int = 400,
                 hop_length: int = 160):
        super().__init__()
        self.libri_ds = libri_ds
        self.mel_transform = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )

    def _log_mel(self, audio: torch.Tensor):
        mel = self.mel_transform(audio)
        log_mel = torch.clamp(mel, min=1e-10).log()
        return log_mel

    def __len__(self):
        return len(self.libri_ds)

    def __getitem__(self, index):
        item = self.libri_ds[index]
        log_mel = self._log_mel(item['audio'].flatten())
        return {
            'log-mel': log_mel,
            'text': item['text']
        }
