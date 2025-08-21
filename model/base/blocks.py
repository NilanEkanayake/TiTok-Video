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

from einops.layers.torch import Rearrange
from einops import rearrange
import math


def get_model_dims(model_size='tiny', head_dim=64, mlp_ratio=4.0):
    if model_size.endswith('_thin'): # https://arxiv.org/pdf/2505.20802
        model_size = model_size[:-5]
        layers = {
            "tiny": 2,
            "small": 5,
            "base": 7,
            "large": 8,
        }[model_size]
        heads = {
            "tiny": [8, 2], # Q heads, KV heads | GQA
            "small": [12, 4],
            "base": [16, 4],
            "large": [32, 8],
        }[model_size]
        mlp_ratio = mlp_ratio/2
    else:
        layers = {
            "tiny": 4,
            "small": 8,
            "base": 12,
            "large": 24,
        }[model_size]
        heads = {
            "tiny": [4, 2],
            "small": [8, 2],
            "base": [12, 4],
            "large": [16, 4],
        }[model_size]

    width = int(head_dim*heads[0])

    return width, layers, heads, mlp_ratio


def init_weights(module):
    if isinstance(module, nn.Linear): # SNLinear has internal init.
        module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
        if module.weight is not None:
            nn.init.constant_(module.weight, 1.0)
    elif isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)
        
    
class TiTokEncoder(nn.Module):
    def __init__(
            self,
            model_size="tiny",
            patch_size=(4, 8, 8),
            in_channels=3,
            out_channels=5,
            max_grid=(16, 64, 64),
            max_tokens=2048,
        ):
        super().__init__()
        self.patch_size = patch_size
        self.token_size = out_channels
        self.in_channels = in_channels

        self.width, self.num_layers, self.heads, mlp_ratio = get_model_dims(model_size)
        scale = self.width ** -0.5

        self.rope = RoPE(
            head_dim=self.width//self.heads[0],
            max_grid=[x//y for x, y in zip(max_grid, patch_size)],
            max_tokens=max_tokens,
        )

        self.mask_token = nn.Parameter(scale * torch.randn(1, self.width)) # LC
        self.proj_in = nn.Linear(in_features=in_channels*math.prod(patch_size), out_features=self.width)

        self.model_layers = ResidualAttentionBlock(
            embed_dim=self.width,
            heads=self.heads,
            mlp_ratio=mlp_ratio,
            num_layer=self.num_layers
        )

        self.ln_post = nn.LayerNorm(self.width)
        self.proj_out = nn.Linear(self.width, self.token_size, bias=True)

    def forward(self, x, token_counts):
        device = x[0].device
        grids = [[dim//ps for dim, ps in zip(vid.shape[1:], self.patch_size)] for vid in x]
        grid_sizes = [math.prod(g) for g in grids]
        seq_lens = [g + t for g, t in zip(grid_sizes, token_counts)]

        # cu_seqlens contains the indices of the packed samples.
        cu_seqlens = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(seq_lens), dim=0)]).to(device, dtype=torch.int32)

        x = [rearrange(vid, 'c (nt pt) (nh ph) (nw pw) -> (nt nh nw) (c pt ph pw)', pt=self.patch_size[0], ph=self.patch_size[1], pw=self.patch_size[2]) for vid in x] # patching
        x = torch.cat(x, dim=0) # LC tensor
        x = self.proj_in(x) # returns LC

        x = torch.split(x, grid_sizes, dim=0)
        masked_tokens = [self.mask_token.repeat(num, 1) for num in token_counts]
        x = [torch.cat([tokens, vid], dim=0) for tokens, vid in zip(masked_tokens, x)]
        x = torch.cat(x, dim=0)

        freqs = self.rope(grids, token_counts, device)
        x = self.model_layers(x, freqs, cu_seqlens, max(seq_lens))

        x = torch.split(x, seq_lens, dim=0)
        x = [tokens[:num] for tokens, num in zip(x, token_counts)]
        x = torch.cat(x, dim=0)

        x = self.ln_post(x)
        x = self.proj_out(x)
        
        return x # packed LC, unpack or leave as-is for FSQ?


class TiTokDecoder(nn.Module):
    def __init__(
            self,
            model_size="tiny",
            patch_size=(4, 8, 8),
            in_channels=5,
            out_channels=3,
            max_grid=(16, 64, 64),
            max_tokens=2048,
        ):
        super().__init__()
        self.patch_size = patch_size
        self.token_size = in_channels
        self.out_channels = out_channels

        self.width, self.num_layers, self.heads, mlp_ratio = get_model_dims(model_size)
        scale = self.width ** -0.5

        self.rope = RoPE(
            head_dim=self.width//self.heads[0],
            max_grid=[x//y for x, y in zip(max_grid, patch_size)],
            max_tokens=max_tokens,
        )

        self.mask_token = nn.Parameter(scale * torch.randn(1, self.width)) # LC
        self.proj_in = nn.Linear(self.token_size, self.width, bias=True)
        self.ln_pre = nn.LayerNorm(self.width)

        self.model_layers = ResidualAttentionBlock(
            embed_dim=self.width,
            heads=self.heads,
            mlp_ratio=mlp_ratio,
            num_layer=self.num_layers
        )

        self.proj_out = nn.Linear(in_features=self.width, out_features=out_channels*math.prod(patch_size))


    def forward(self, x, token_counts, grids):
        device = x.device
        grids = [[dim//ps for dim, ps in zip(grid, self.patch_size)] for grid in grids]
        grid_sizes = [math.prod(grid) for grid in grids]
        seq_lens = [g + t for g, t in zip(grid_sizes, token_counts)]
        cu_seqlens = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(seq_lens), dim=0)]).to(device, dtype=torch.int32)

        x = self.proj_in(x) # LC in?

        x = torch.split(x, token_counts, dim=0)
        mask_tokens = [self.mask_token.repeat(grid_size, 1) for grid_size in grid_sizes]
        x = [torch.cat([tokens, masked_vid], dim=0) for tokens, masked_vid in zip(x, mask_tokens)]
        x = torch.cat(x, dim=0)

        x = self.ln_pre(x)

        freqs = self.rope(grids, token_counts, device)
        x = self.model_layers(x, freqs, cu_seqlens, max(seq_lens))

        x = torch.split(x, seq_lens, dim=0)
        x = [tokens[num:] for tokens, num in zip(x, token_counts)]
        x = torch.cat(x, dim=0)

        x = self.proj_out(x)
        x = torch.split(x, grid_sizes, dim=0)

        x = [rearrange(
                vid,
                '(nt nh nw) (c pt ph pw) -> c (nt pt) (nh ph) (nw pw)',
                nt=grid[0], nh=grid[1], nw=grid[2],
                pt=self.patch_size[0], ph=self.patch_size[1], pw=self.patch_size[2],
            ) for vid, grid in zip(x, grids)
        ]
        
        return x # list of video tensors in (CTHW) out