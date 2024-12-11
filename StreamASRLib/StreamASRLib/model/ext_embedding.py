import torch
import torch.nn as nn


class ExtendedEmbedding(nn.Module):
    def __init__(self, base_emb: nn.Embedding, ext_tokens_size: int, ext_token_threshold: int):
        super().__init__()
        self.base_emb = base_emb
        self.hidden_size = base_emb.embedding_dim
        self.ext_emb = nn.Embedding(ext_tokens_size, self.hidden_size)
        self.ext_token_threshold = ext_token_threshold

    def freeze_base_emb(self):
        for param in self.base_emb.parameters():
            param.requires_grad = False

    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        ext_token_mask = (input_tokens >= self.ext_token_threshold)

        inp_emb = torch.zeros(list(input_tokens.shape) +
                              [self.hidden_size]).to(input_tokens.device)
        inp_emb[~ext_token_mask] = self.base_emb(input_tokens[~ext_token_mask])
        inp_emb[ext_token_mask] = self.ext_emb(
            input_tokens[ext_token_mask] - self.ext_token_threshold)
        return inp_emb
