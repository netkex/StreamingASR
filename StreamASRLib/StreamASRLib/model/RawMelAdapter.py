from typing import Optional

import torch
import torch.nn.functional as F
import torch.nn as nn
from torchtune.modules import RotaryPositionalEmbeddings

from .blocks import AttentionWithRope


class RawMelAdapter(nn.Module):
    def __init__(self,
                 n_mels: int,
                 hidden_size: int,
                 attention_layers: int,
                 attn_num_heads: int = 4,
                 attn_ffn_neck: int = 1024,
                 proj_size: Optional[int] = None,
                 add_rope: bool = True):
        super().__init__()
        if proj_size is None:
            proj_size = hidden_size

        self.conv1 = nn.Conv1d(n_mels, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size,
                               kernel_size=3, stride=2, padding=1)

        self.rope = RotaryPositionalEmbeddings(
            hidden_size // attn_num_heads) if add_rope else None
        self.attention = nn.ModuleList(
            [AttentionWithRope(
                emb_dim=hidden_size,
                hidden_dim=hidden_size,
                num_heads=attn_num_heads,
                ffn_neck=attn_ffn_neck)
             for _ in range(attention_layers)]
        )

        self.proj = nn.Linear(hidden_size, proj_size)

    @staticmethod
    def _length_after_conv(conv: nn.Conv1d, input_length: torch.Tensor) -> torch.Tensor:
        return torch.floor(
            (input_length + 2 * conv.padding[0] - conv.dilation[0] * (conv.kernel_size[0] - 1) - 1) / conv.stride[0] + 1)

    def get_length(self, input_length: torch.Tensor) -> torch.Tensor:
        after_conv1 = RawMelAdapter._length_after_conv(
            self.conv1, input_length)
        after_conv2 = RawMelAdapter._length_after_conv(
            self.conv2, after_conv1)
        return after_conv2.long()

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.conv1(input))
        x = F.gelu(self.conv2(x))
        x = x.transpose(-1, -2)

        for attn in self.attention:
            x = attn(x, rope=self.rope)
        return self.proj(x)
