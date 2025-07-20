"""Modified from: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_ltx.py"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nested import nested_tensor_from_jagged
from einops import einsum, rearrange, reduce
from typing import Any, Dict, Optional, Tuple, Union
import math


def apply_rotary_emb(x, freqs):
    dtype = x.dtype
    cos, sin = freqs
    x = x.float()

    x_real, x_imag = x.chunk(2, dim=-1)
    x_rotated = torch.cat([-x_imag, x_real], dim=-1)
    out = (x * cos + x_rotated * sin)
    return out.to(dtype)


class RoPE(nn.Module):
    def __init__(
            self,
            dim=512,
            theta=10000.0,
            # max_grid=[16, 168, 168],
            # patch_size=[4, 8, 8],
        ):
        super(RoPE, self).__init__()
        self.dim = dim
        self.theta = theta

        # self.base_grid = [x//y for x, y in zip(max_grid, patch_size)]
        # no caching/prefill yet

    def compute_freqs(
        self,
        in_grid: Tuple[int, ...],
        device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        coords = [torch.arange(size, dtype=torch.float32, device=device) for size in in_grid] # [[T], [H], [W]] for 3D
        grid = torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=0) # [THW]

        grid = grid.repeat(1, *(1 for _ in in_grid)) # C*

        # if len(in_grid) == len(self.base_grid): # clunky. Also add for 1D as well?
        #     # applies along the 1-size channel dim?
        #     grid[0:1] = grid[0:1] / self.base_grid[0] # T
        #     grid[1:2] = grid[1:2] / self.base_grid[1] # H
        #     grid[2:3] = grid[2:3] / self.base_grid[2] # W

        grid = grid.flatten(start_dim=1).transpose(0, 1) # convert to LC

        start = 1.0
        end = self.theta
        freqs = self.theta ** torch.linspace(
            math.log(start, self.theta), # start
            math.log(end, self.theta), # stop
            self.dim // (2 * len(in_grid)), # steps (size of tensor) -> for patches: 512/(2*3) = 85 (then padded) -> thus produces [85] as out tensor shape
            device=device,
            dtype=torch.float32,
        )

        freqs = freqs * math.pi / 2.0 # value adjust

        freqs = freqs * (grid.unsqueeze(-1) * 2 - 1) # LC -> LCN
        freqs = freqs.transpose(-1, -2).flatten(1) # LCN -> LNC -> LC

        cos_freqs = freqs.cos().repeat_interleave(2, dim=-1)
        sin_freqs = freqs.sin().repeat_interleave(2, dim=-1)

        padding_dim = self.dim % (2 * len(in_grid))
        if padding_dim != 0:
            cos_padding = torch.ones_like(cos_freqs[:, : padding_dim])
            sin_padding = torch.zeros_like(cos_freqs[:, : padding_dim])
            cos_freqs = torch.cat([cos_padding, cos_freqs], dim=-1)
            sin_freqs = torch.cat([sin_padding, sin_freqs], dim=-1)

        return cos_freqs, sin_freqs

    def forward(self, x, grids, token_counts):
        device = x.device
        offsets = x.offsets()

        with torch.autocast(device.type, enabled=False):
            freqs_patches = []
            freqs_tokens = []

            for grid, token_count in zip(grids, token_counts):
                freqs_patches.append(self.compute_freqs(grid, device))
                freqs_tokens.append(self.compute_freqs([token_count], device))

            cos = [torch.cat([f_p[0], f_t[0]], dim=0) for f_p, f_t in zip(freqs_patches, freqs_tokens)]
            sin = [torch.cat([f_p[1], f_t[1]], dim=0) for f_p, f_t in zip(freqs_patches, freqs_tokens)]
            cos = nested_tensor_from_jagged(values=torch.cat(cos, dim=0), offsets=offsets, jagged_dim=1)
            sin = nested_tensor_from_jagged(values=torch.cat(sin, dim=0), offsets=offsets, jagged_dim=1)

        return cos, sin