import torch
import torchaudio
from StreamASRLib.model.config import DEVICE


class RawLibriSpeechPT(torch.utils.data.Dataset):
    def __init__(self,
                 split="test-clean",
                 path='/tmp/librispeech',
                 device=DEVICE):
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=path,
            url=split,
            download=True
        )
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        audio, sample_rate, text, _, _, _ = self.dataset[item]
        return {
            'audio': audio.to(self.device),
            'text': text,
            'sample_rate': sample_rate
        }


class PadTrimCollator:
    def __init__(self, pad_value=0, max_len=None, dim=-1):
        self.pad_value = pad_value
        self.dim = dim
        self.max_len = max_len

    @staticmethod
    def pad_or_trim_tensor(tensor, length, pad_value, dim=-1):
        if tensor.shape[dim] > length:
            return tensor.index_select(index=torch.arange(length, device=tensor.device), dim=dim)
        if tensor.shape[dim] < length:
            pad_shape = list(tensor.shape)
            pad_shape[dim] = length - tensor.shape[dim]
            return torch.cat((tensor, torch.full(size=pad_shape, fill_value=pad_value, device=tensor.device)), dim=dim)
        return tensor

    def __pad_collate(self, batch):
        max_len = max(map(lambda item: item[0].shape[self.dim], batch))
        if self.max_len is not None:
            max_len = min(self.max_len, max_len)
        batched_tensors = torch.stack(
            list(map(
                lambda item: PadTrimCollator.pad_or_trim_tensor(
                    item[0], max_len, self.pad_value, self.dim),
                batch
            )),
            dim=0
        )
        rest_batch = []
        for i in range(1, len(batch[0])):
            rest_batch.append(list(map(lambda item: item[i], batch)))
        return (batched_tensors, *rest_batch)

    def __call__(self, batch):
        return self.__pad_collate(batch)
