import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings


class AttentionBlockWithRope(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int, num_heads: int = 4, bottle_neck: int = 64):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.k_proj = nn.Linear(emb_dim, hidden_dim)
        self.q_proj = nn.Linear(emb_dim, hidden_dim)
        self.v_proj = nn.Linear(emb_dim, hidden_dim)
        self.multi_head_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True)

        self.out_proj = nn.Linear(
            hidden_dim, emb_dim) if hidden_dim != emb_dim else nn.Identity()
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, bottle_neck),
            nn.Linear(bottle_neck, emb_dim)
        )

        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, input: torch.Tensor, rope: RotaryPositionalEmbeddings = None, attn_mask: torch.Tensor = None) -> torch.Tensor:
        key = self.k_proj(input)
        query = self.q_proj(input)
        value = self.v_proj(input)
        if rope is not None:
            query_ = query.reshape(
                list(query.shape[:-1]) + [self.num_heads, self.head_dim])
            key_ = key.reshape(
                list(key.shape[:-1]) + [self.num_heads, self.head_dim])
            rot_query = rope(query_)
            rot_key = rope(key_)
            query = rot_query.reshape(query.shape)
            key = rot_key.reshape(key.shape)

        attn, _ = self.multi_head_attn(
            query, key, value, need_weights=False, attn_mask=attn_mask)
        ffn_attn = self.ffn(self.out_proj(attn))
        res = self.norm(input + ffn_attn)
        return res


class WavTokensEncoder(nn.Module):
    def __init__(self,
                 num_tokens: int,
                 hidden_size: int,
                 num_layers: int,
                 num_heads: int = 4,
                 add_rope: bool = True,
                 proj_size: int = None):
        nn.Transformer
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, hidden_size)

        self.num_layers = num_layers
        self.rope = RotaryPositionalEmbeddings(
            hidden_size // num_heads) if add_rope else None
        self.encoder = nn.ModuleList([
            AttentionBlockWithRope(hidden_size, hidden_size, num_heads)
            for _ in range(num_layers)
        ])
        self.proj = nn.Linear(
            hidden_size, proj_size) if proj_size is not None else nn.Identity()

    def forward(self, input: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        if attn_mask is None:
            attn_mask = torch.full((input.shape[-1], input.shape[-1]), True)

        x = self.embedding(input)
        for l_id in range(self.num_layers):
            x = self.encoder[l_id](x, rope=self.rope, attn_mask=attn_mask)
        return self.proj(x)

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, num_tokens: int, hidden_size: int, num_layers: int, proj_size: int = None):
        model = cls(num_tokens, hidden_size, num_layers, proj_size)
        model.load_state_dict(torch.load(path, weights_only=True))
