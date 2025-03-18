import torch


def build_ref_seq(tokenizer, output: torch.Tensor) -> str:
    tokens = torch.argmax(output, dim=-1)
    return tokenizer.decode(tokens.cpu())
