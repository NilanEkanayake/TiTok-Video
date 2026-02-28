"""Building blocks for TiTok.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 

Reference: 
    https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transformer.py
    https://github.com/baofff/U-ViT/blob/main/libs/timm.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base.transformer import ResidualAttentionBlock
from model.base.rope import RoPE
from model.base.utils import get_model_dims, patch_rearrange, unpatch_rearrange
from flash_attn.ops.triton.layer_norm import RMSNorm
import math
        
    
class TiTokEncoder(nn.Module):
    def __init__(
            self,
            model_size="tiny",
            patch_size=(4, 8, 8),
            in_channels=3,
            out_channels=5,
        ):
        super().__init__()
        # self.patch_size = patch_size
        self.patch_size = torch.tensor(patch_size, dtype=torch.int32)
        self.token_size = out_channels
        self.in_channels = in_channels

        self.width, self.num_layers, self.heads, mlp_ratio = get_model_dims(model_size)
        scale = self.width ** -0.5

        self.patching = patch_rearrange(patch_size)
        self.proj_in = nn.Linear(in_features=in_channels*math.prod(patch_size), out_features=self.width)
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1))
        self.ln_pre_t = RMSNorm(self.width)
        self.ln_pre_p = RMSNorm(self.width)

        self.rope = RoPE(
            head_dim=self.width//self.heads[0],
            grid_dims=len(patch_size),
        )

        self.model_layers = ResidualAttentionBlock(
            embed_dim=self.width,
            heads=self.heads,
            mlp_ratio=mlp_ratio,
            num_layer=self.num_layers
        )

        self.ln_post = RMSNorm(self.width)
        self.proj_out = nn.Linear(self.width, self.token_size, bias=True)


    # @torch.compile()
    def forward(self, videos, token_counts, grids=None): # X = list of CHW?
        B = len(token_counts)
        device = videos[0].device
        dtype = videos[0].dtype

        ###
        if grids is None:
            grids = torch.tensor([v.shape[1:] for v in videos], device=device, dtype=torch.int32)

        grids = grids // self.patch_size.to(grids).unsqueeze(0)
        grid_sizes = grids.prod(dim=-1)
        seq_lens = grid_sizes + token_counts # elementwise sum
        cu_seqlens = torch.cat([torch.tensor([0], device=device), seq_lens.cumsum(0)]).to(torch.int32)

        values = torch.tensor([1, 0], device=device, dtype=torch.bool).repeat(B)
        mask = torch.repeat_interleave(values, torch.stack([token_counts, grid_sizes], dim=-1).flatten())

        freqs = self.rope(grids, token_counts, device)
        ###

        patches = [self.patching(v) for v in videos]
        patches = torch.cat(patches, dim=0) # LC tensor
        patches = self.proj_in(patches)

        x = torch.zeros([mask.shape[0], self.width], device=device, dtype=dtype)
        x[mask] = self.ln_pre_t(self.mask_token.to(dtype).expand(-1, self.width))
        x[~mask] = self.ln_pre_p(patches + self.mask_token.to(dtype)) # apply split norm here?

        x = self.model_layers(x, freqs, cu_seqlens, seq_lens.max())

        tokens = x[mask].contiguous() # get tokens
        tokens = self.ln_post(tokens)
        tokens = self.proj_out(tokens)
        return tokens



class TiTokDecoder(nn.Module):
    def __init__(
            self,
            model_size="tiny",
            patch_size=(4, 8, 8),
            in_channels=5,
            out_channels=3,
        ):
        super().__init__()
        # self.patch_size = patch_size
        self.patch_size = torch.tensor(patch_size, dtype=torch.int32)
        self.token_size = in_channels
        self.out_channels = out_channels

        self.width, self.num_layers, self.heads, mlp_ratio = get_model_dims(model_size)
        scale = self.width ** -0.5

        self.proj_in = nn.Linear(self.token_size, self.width, bias=True)
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1))
        self.ln_pre_t = RMSNorm(self.width)
        self.ln_pre_p = RMSNorm(self.width)

        self.rope = RoPE(
            head_dim=self.width//self.heads[0],
            grid_dims=len(patch_size),
        )

        self.model_layers = ResidualAttentionBlock(
            embed_dim=self.width,
            heads=self.heads,
            mlp_ratio=mlp_ratio,
            num_layer=self.num_layers
        )

        self.ln_post = RMSNorm(self.width)
        self.proj_out = nn.Linear(in_features=self.width, out_features=out_channels*math.prod(patch_size))
        self.unpatching = unpatch_rearrange(patch_size)


    # @torch.compile()
    def forward(self, tokens, token_counts, grids):
        B = len(token_counts)
        device = tokens.device
        dtype = tokens.dtype

        ###
        grids = grids // self.patch_size.to(grids).unsqueeze(0)
        grid_sizes = grids.prod(dim=-1)
        seq_lens = grid_sizes + token_counts # elementwise sum
        cu_seqlens = torch.cat([torch.tensor([0], device=device), seq_lens.cumsum(0)]).to(torch.int32)

        values = torch.tensor([1, 0], device=device, dtype=torch.bool).repeat(B)
        mask = torch.repeat_interleave(values, torch.stack([token_counts, grid_sizes], dim=-1).flatten())
        
        freqs = self.rope(grids, token_counts, device)
        ###

        x = torch.zeros([mask.shape[0], self.width], device=device, dtype=dtype)
        x[mask] = self.ln_pre_t(self.proj_in(tokens) + self.mask_token.to(dtype))
        x[~mask] = self.ln_pre_p(self.mask_token.to(dtype).expand(-1, self.width))

        x = self.model_layers(x, freqs, cu_seqlens, seq_lens.max())

        patches = x[~mask].contiguous() # get patches
        patches = self.ln_post(patches)
        patches = self.proj_out(patches)
        patches = torch.split(patches, grid_sizes.tolist(), dim=0)

        videos = [self.unpatching(p, g) for p, g in zip(patches, grids)]
        return videos # list of video tensors in (CTHW) out
    

# tests here
if __name__ == '__main__':
    import random
    import time

    B = 16
    MAX_GRID = [128, 128]
    PATCH_SIZE = [8, 8]
    MAX_TL = 256

    device = 'cuda:0'
    dtype = torch.bfloat16

    model = TiTokEncoder().to(device, dtype)

    x = [torch.rand([3] + [random.randrange(PATCH_SIZE[i], MAX_GRID[i]+1, step=PATCH_SIZE[i]) for i in range(len(MAX_GRID))]).to(device, dtype) for _ in range(B)]
    token_counts = [random.randrange(1, MAX_TL+1) for _ in range(B)]

    start_t = time.time()
    out_fast = model.forward(x, token_counts)
    fast_t = time.time() - start_t

    start_t = time.time()
    out_fast = model.forward(x, token_counts)
    fast_t_2 = time.time() - start_t

    # assert torch.equal(out_norm, out_fast)
    # print(norm_t)
    print(fast_t)
    print(fast_t_2)
