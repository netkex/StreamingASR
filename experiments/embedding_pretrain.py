import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseAttention(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int):
        super().__init__()
        self.k_proj = nn.Linear(emb_dim, hidden_dim)
        self.q_proj = nn.Linear(emb_dim, hidden_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, input, attn_mask=None):
        key = self.k_proj(input)
        query = self.q_proj(input)
        value = self.v_proj(input)

        attn_emb = F.scaled_dot_product_attention(query, key, value, attn_mask)
        res = self.norm(input + attn_emb)
        return res


class EmbeddingBaseModel(nn.Module):
    def __init__(self, num_tokens: int, hidden_dim: int):
        super().__init__()
        self.emb = nn.Embedding(num_tokens, hidden_dim)
        self.attn = nn.ModuleList([
            BaseAttention(hidden_dim, hidden_dim),
            BaseAttention(hidden_dim, hidden_dim)
        ])
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, input, attn_mask=None):
        x = self.emb(input)

        for attn in self.attn:
            x = attn(x, attn_mask)
        return self.proj(x[..., 0, :])


model = EmbeddingBaseModel(4096, 768)
criterion = nn.CosineEmbeddingLoss()
batch = torch.tensor(np.random.choice(np.arange(100), (4, 100)))
model(batch)
