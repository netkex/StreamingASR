import torch


class PadTrimDictCollator:
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
        assert len(batch) != 0

        res_batch = {}
        for key in batch[0].keys():
            if type(batch[0][key]) == torch.Tensor:
                max_len = max(
                    map(lambda item: item[key].shape[self.dim], batch))
                if self.max_len is not None:
                    max_len = min(self.max_len, max_len)
                batched_tensors = torch.stack(
                    list(map(
                        lambda item: PadTrimDictCollator.pad_or_trim_tensor(
                            item[key], max_len, self.pad_value, self.dim),
                        batch
                    )),
                    dim=0
                )
                res_batch[key] = batched_tensors
            else:
                values = []
                for item in batch:
                    values.append(item[key])
                res_batch[key] = values
        return res_batch

    def __call__(self, batch):
        return self.__pad_collate(batch)
