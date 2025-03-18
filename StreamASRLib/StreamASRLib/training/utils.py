import torch
import json
from munch import Munch


def build_ref_seq(tokenizer, output: torch.Tensor) -> str:
    tokens = torch.argmax(output, dim=-1)
    return tokenizer.decode(tokens.cpu())


def read_config(path: str):
    with open(path, "r") as config_file:
        return Munch.fromDict(json.load(config_file))
