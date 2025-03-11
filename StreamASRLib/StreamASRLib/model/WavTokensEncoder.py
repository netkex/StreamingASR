import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int):
        super().__init__()
        self.k_proj = nn.Linear(emb_dim, hidden_dim)
        self.q_proj = nn.Linear(emb_dim, hidden_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, input: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        key = self.k_proj(input)
        query = self.q_proj(input)
        value = self.v_proj(input)
        attn_emb = F.scaled_dot_product_attention(query, key, value, attn_mask)
        res = self.norm(input + attn_emb)
        return res


class WavTokensEncoder(nn.Module):
    def __init__(self, num_tokens: int, hidden_size: int, num_layers: int, proj_size: int = None):
        nn.Transformer
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, hidden_size)

        self.num_layers = num_layers
        self.encoder = nn.ModuleList([
            AttentionBlock(hidden_size, hidden_size)
            for _ in range(num_layers)
        ])
        self.proj = nn.Linear(
            hidden_size, proj_size) if proj_size is not None else nn.Identity()

    def forward(self, input: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        if attn_mask is None:
            attn_mask = torch.full((input.shape[-1], input.shape[-1]), True)

        x = self.embedding(input)
        for l_id in range(self.num_layers):
            x = self.encoder[l_id](x, attn_mask)
        return self.proj(x)

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, num_tokens: int, hidden_size: int, num_layers: int, proj_size: int = None):
        model = cls(num_tokens, hidden_size, num_layers, proj_size)
        model.load_state_dict(torch.load(path, weights_only=True))
