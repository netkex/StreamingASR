import torch
import torch.nn as nn
from torchtune.modules import RotaryPositionalEmbeddings


class AttentionWithRope(nn.Module):
    def __init__(self,
                 emb_dim: int,
                 hidden_dim: int,
                 num_heads: int = 4,
                 ffn_neck: int = 1024,
                 is_causal: bool = True):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.is_causal = is_causal

        self.k_proj = nn.Linear(emb_dim, hidden_dim)
        self.q_proj = nn.Linear(emb_dim, hidden_dim)
        self.v_proj = nn.Linear(emb_dim, hidden_dim)
        self.multi_head_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True)

        self.out_proj = nn.Linear(
            hidden_dim, emb_dim) if hidden_dim != emb_dim else nn.Identity()
        self.attn_ffn = nn.Sequential(
            nn.Linear(emb_dim, ffn_neck),
            nn.ReLU(),
            nn.Linear(ffn_neck, emb_dim)
        )
        self.attn_norm = nn.RMSNorm(emb_dim)

        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, ffn_neck),
            nn.ReLU(),
            nn.Linear(ffn_neck, emb_dim)
        )
        self.norm = nn.RMSNorm(emb_dim)

    def forward(self, input: torch.Tensor, rope: RotaryPositionalEmbeddings = None) -> torch.Tensor:
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

        if self.is_causal:
            attn_mask = nn.Transformer.generate_square_subsequent_mask(
                key.shape[-2], key.device)
            attn, _ = self.multi_head_attn(
                query, key, value, need_weights=False, attn_mask=attn_mask, is_causal=True)
        else:
            attn, _ = self.multi_head_attn(
                query, key, value, need_weights=False, is_causal=False)

        x = self.attn_norm(input + self.attn_ffn(attn))
        return self.norm(x + self.ffn(x))
