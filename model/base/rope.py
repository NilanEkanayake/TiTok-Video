import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, reduce
from typing import Tuple, Union, List
import math
import numpy as np


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    freqs_dtype=torch.float32,  #  torch.float32, torch.float64 (flux)
):
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)  # type: ignore  # [S]

    freqs = (1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim)))  # [D/2]
    freqs = torch.outer(pos, freqs)  # type: ignore   # [S, D/2]

    # flux, hunyuan-dit, cogvideox
    freqs_cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
    freqs_sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
    return freqs_cos, freqs_sin
    

def apply_rotary_emb(x, freqs):
    dtype = x.dtype
    cos, sin = freqs

    with torch.autocast(x.device.type, enabled=False):
        x = x.contiguous().float()
        cos = cos.contiguous().float().unsqueeze(-2) # unsqueeze head dim
        sin = sin.contiguous().float().unsqueeze(-2)
        
        x_real, x_imag = x.chunk(2, dim=-1)
        x_rotated = torch.cat([-x_imag, x_real], dim=-1).contiguous().float()
        out = (x * cos + x_rotated * sin)
        out = out.to(dtype)

    return out

class RotaryPosEmbed(nn.Module):
    def __init__(
        self,
        axes_dim: List[int] = (20, 22, 22),
        theta: float = 10000.0,
    ) -> None:
        super().__init__()

        self.theta = theta
        self.axes_dim = axes_dim


    def compute_grid(self, in_grid, num_tokens, device):
        frames, height, width = in_grid
        seq_len = math.prod(in_grid) + num_tokens # same as max_seq_len

        # Create position IDs -> [L, 3], all zeros. 3 = [frames, height, width]. Tokens are packed into THW dims like orig m-rope.
        position_ids = torch.zeros(seq_len, len(in_grid), dtype=torch.float32, device=device)
        position_ids[:num_tokens] = torch.arange(num_tokens, dtype=torch.float32, device=device).unsqueeze(-1) # assign to THW dims.

        # add THW position ids
        position_ids[num_tokens:, 0] = ( # frames
            torch.arange(frames, dtype=torch.float64, device=device)
            .view(-1, 1, 1)
            .repeat(1, height, width)
            .flatten()
        )

        position_ids[num_tokens:, 1] = ( # height
            torch.arange(height, dtype=torch.float64, device=device)
            .view(1, -1, 1)
            .repeat(frames, 1, width)
            .flatten()
        )

        position_ids[num_tokens:, 2] = ( # width
            torch.arange(width, dtype=torch.float64, device=device)
            .view(1, 1, -1)
            .repeat(frames, height, 1)
            .flatten()
        )

        position_ids[num_tokens:] += num_tokens

        return position_ids


    def forward(
        self,
        in_grid: Tuple[int, ...],
        num_tokens: int,
        device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
                
        grid = self.compute_grid(in_grid, num_tokens, device)

        # freqs_t = get_1d_rotary_pos_embed(self.axes_dim[0], grid[:, 0], theta=self.theta, freqs_dtype=torch.float64)
        # freqs_h = get_1d_rotary_pos_embed(self.axes_dim[1], grid[:, 1], theta=self.theta, freqs_dtype=torch.float64)
        # freqs_w = get_1d_rotary_pos_embed(self.axes_dim[2], grid[:, 2], theta=self.theta, freqs_dtype=torch.float64)

        # cos_freqs = torch.cat([freqs_t[0], freqs_h[0], freqs_w[0]], dim=-1)
        # sin_freqs = torch.cat([freqs_t[1], freqs_h[1], freqs_w[1]], dim=-1)

        start = 1.0
        end = self.theta
        freqs = self.theta ** torch.linspace(
            math.log(start, self.theta), # 0.0
            math.log(end, self.theta), # 1.0
            sum(self.axes_dim) // (2 * grid.shape[-1]),
            device=device,
            dtype=torch.float64,
        )
        freqs = freqs * math.pi / 2.0
        freqs = freqs * grid.unsqueeze(-1)
        freqs = freqs.transpose(-1, -2).flatten(1)
        
        cos_freqs = freqs.cos().repeat_interleave(2, dim=-1)
        sin_freqs = freqs.sin().repeat_interleave(2, dim=-1)

        padding_dim = sum(self.axes_dim) % (2 * grid.shape[-1])
        if padding_dim != 0:
            cos_padding = torch.ones_like(cos_freqs[:, : padding_dim])
            sin_padding = torch.zeros_like(cos_freqs[:, : padding_dim])
            cos_freqs = torch.cat([cos_padding, cos_freqs], dim=-1)
            sin_freqs = torch.cat([sin_padding, sin_freqs], dim=-1)

        return cos_freqs, sin_freqs


class RoPE(nn.Module):
    def __init__(
            self,
            dim=None,
        ):
        super(RoPE, self).__init__()
        self.pos_emb = RotaryPosEmbed()
        
    
    def forward(self, grids, token_counts, device):
        with torch.autocast(device.type, enabled=False):
            all_cos = []
            all_sin = []
            for grid, token_count in zip(grids, token_counts):
                cos, sin = self.pos_emb(grid, token_count, device)
                all_cos.append(cos)
                all_sin.append(sin)

            cos = torch.cat(all_cos, dim=0) # [B*L, C]
            sin = torch.cat(all_sin, dim=0)

        return cos, sin
    

if __name__ == '__main__':
    B = 1
    D = 512
    H = 8
    
    IN_GRID = (2, 4, 4) # T, H, W
    NUM_TOKENS = 16
    SEQ_LEN = math.prod(IN_GRID) + NUM_TOKENS

    rope = RoPE(D)

    x = torch.randn(B*SEQ_LEN, H, D//H)
    freqs = rope([IN_GRID], [NUM_TOKENS], x.device)

    print(x.shape)
    print(freqs[0].shape)

    x = apply_rotary_emb(x, freqs)
    print(x.shape)
