import torch
from torch.utils.data import Dataset


class MergeDataset:
    def __init__(self, datasets):
        self.datasets = datasets

    def __len__(self):
        return sum(map(len, self.datasets))

    def __getitem__(self, id):
        for ds_id in range(len(self.datasets)):
            if len(self.datasets[ds_id]) <= id:
                id -= len(self.datasets[ds_id])
                continue
            return self.datasets[ds_id][id]


class LibriDataset:
    def __init__(self, librispeech_ds):
        self.librispeech_ds = librispeech_ds

    def __len__(self):
        return len(self.librispeech_ds)

    def __getitem__(self, id):
        audio_item = self.librispeech_ds[id]
        return {
            'audio': audio_item[0],
            'sr': audio_item[1],
            'text': audio_item[2]
        }
