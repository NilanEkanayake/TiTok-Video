import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from einops import einsum, rearrange, reduce

"""
References:
https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_ltx.py
https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_lumina2.py
https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/embeddings.py
"""

def apply_rotary_emb(x, freqs_cis):
    with torch.autocast(x.device.type, enabled=False):
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(-2) # unsqueeze head dim -> [L, H, D]
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(-2)

    return x_out.type_as(x)


def get_1d_rotary_pos_embed(dim, pos, theta=10000.0, freqs_dtype=torch.float64): 
    assert dim % 2 == 0

    if type(pos) is int:
        pos = torch.arange(pos)

    start = 1.0
    end = theta

    freqs = theta ** torch.linspace(
        math.log(start, theta), # 0.0?
        math.log(end, theta), # 1.0?
        dim//2,
        device=pos.device,
        dtype=freqs_dtype,
    )
    freqs = freqs * math.pi / 2.0
    freqs = freqs * pos.unsqueeze(-1)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis
    

class Lumina2RotaryPosEmbed(nn.Module):
    def __init__(
            self,
            theta: float = 10000.0,
            axes_dim: List[int] = (24, 20, 20),
            axes_lens: Optional[List[int]] = None,
        ):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.axes_lens = axes_lens

        if self.axes_lens is not None:
            self.freqs_cis = self._precompute_freqs_cis(axes_dim, axes_lens, theta)

    def _precompute_freqs_cis(self, axes_dim: List[int], axes_lens: List[int], theta: int) -> List[torch.Tensor]:
        freqs_cis = []
        for d, e in zip(axes_dim, axes_lens):
            emb = get_1d_rotary_pos_embed(d, e, theta=theta)
            freqs_cis.append(emb)
        return freqs_cis

    def _get_precomputed_freqs_cis(self, ids: torch.Tensor) -> torch.Tensor:
        device = ids.device

        result = []
        for i in range(len(self.axes_dim)):
            freqs = self.freqs_cis[i].to(ids.device)
            index = ids[:, i : i + 1].repeat(1, freqs.shape[-1])
            result.append(torch.gather(freqs, dim=0, index=index))
        return torch.cat(result, dim=-1).to(device)
    
    def _get_freqs_cis(self, ids: torch.Tensor) -> torch.Tensor:
        device = ids.device

        result = []
        for i in range(len(self.axes_dim)):
            freqs = get_1d_rotary_pos_embed(self.axes_dim[i], ids[:, i], theta=self.theta).to(ids.device)
            result.append(freqs)
        return torch.cat(result, dim=-1).to(device)


    def forward(self, in_grid: Tuple[int, int, int], num_tokens: int, device: str):
        frames, height, width = in_grid
        seq_len = math.prod(in_grid) + num_tokens

        # Create position IDs -> [L, 3], all zeros. 3 = [frames, height, width]. Tokens are packed into THW dims like orig m-rope.
        position_ids = torch.zeros(seq_len, len(in_grid), dtype=torch.int64, device=device)
        position_ids[:num_tokens] = torch.arange(num_tokens, dtype=torch.int64, device=device).unsqueeze(-1) # assign to THW dims.

        # add THW position ids
        position_ids[num_tokens:, 0] = ( # frames
            torch.arange(frames, dtype=torch.int64, device=device)
            .view(-1, 1, 1)
            .repeat(1, height, width)
            .flatten()
        )

        position_ids[num_tokens:, 1] = ( # height
            torch.arange(height, dtype=torch.int64, device=device)
            .view(1, -1, 1)
            .repeat(frames, 1, width)
            .flatten()
        )

        position_ids[num_tokens:, 2] = ( # width
            torch.arange(width, dtype=torch.int64, device=device)
            .view(1, 1, -1)
            .repeat(frames, height, 1)
            .flatten()
        )

        position_ids[num_tokens:] += num_tokens # offset THW to increment from 1D enc

        if self.axes_lens is not None and self.training: # instead, check if in_grid less than the max precomputed/axes lens?
            freqs_cis = self._get_precomputed_freqs_cis(position_ids) # use precomputed
        else:
            freqs_cis = self._get_freqs_cis(position_ids)

        return freqs_cis


class RoPE(nn.Module):
    def __init__(
            self,
            theta=10000.0,
            head_dim=64,
            max_grid=[16, 64, 64],
            max_tokens=2048,
        ):
        super(RoPE, self).__init__()
        axes_dim = head_dim/len(max_grid)
        axes_dim = [int(axes_dim - (axes_dim % 2))] * len(max_grid)
        axes_dim[0] += head_dim - sum(axes_dim) # add remainder to T dim

        axes_lens = [x + max_tokens for x in max_grid]

        self.pos_emb = Lumina2RotaryPosEmbed(theta, axes_dim, axes_lens)


    def forward(self, grids, token_counts, device):
        with torch.autocast(device.type, enabled=False):
            freqs = []
            for grid, token_count in zip(grids, token_counts):
                freqs.append(self.pos_emb(grid, token_count, device))

            freqs = torch.cat(freqs, dim=0) # [B*L, C]

        return freqs
    

if __name__ == '__main__':
    B = 1
    D = 512
    H = 8

    # 
    
    IN_GRID = (2, 4, 4) # T, H, W
    NUM_TOKENS = 16
    SEQ_LEN = (math.prod(IN_GRID) + NUM_TOKENS) * B

    rope = RoPE()

    x = torch.randn(SEQ_LEN, H, D//H) # LHD

    freqs = rope([IN_GRID], [NUM_TOKENS], x.device) # torch.Size([1, 8, 48, 64])
    print(freqs.shape)

    x = apply_rotary_emb(x, freqs)
    print(x.shape)