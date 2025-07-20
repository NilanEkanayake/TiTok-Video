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
from model.base.sigma_reparam import SNLinear
from model.base.rope import RoPE

from torch.nested import nested_tensor, as_nested_tensor

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
            "tiny": 8,
            "small": 12,
            "base": 16,
            "large": 30,
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
            "tiny": 4,
            "small": 8,
            "base": 12,
            "large": 16,
        }[model_size]

    width = int(head_dim*heads)

    return width, layers, heads, mlp_ratio
        
    
class TiTokEncoder(nn.Module):
    def __init__(
            self,
            model_size="tiny",
            patch_size=(4, 8, 8),
            in_channels=3,
            out_channels=5,
        ):
        super().__init__()
        self.patch_size = patch_size
        self.token_size = out_channels
        self.in_channels = in_channels

        self.width, self.num_layers, self.num_heads, mlp_ratio = get_model_dims(model_size)
        scale = self.width ** -0.5

        self.rope = RoPE(self.width)

        self.mask_token = nn.Parameter(scale * torch.randn(1))

        self.proj_in = SNLinear(in_features=in_channels*math.prod(patch_size), out_features=self.width)

        self.model_layers = ResidualAttentionBlock(embed_dim=self.width, num_head=self.num_heads, mlp_ratio=mlp_ratio, num_layer=self.num_layers)

        self.ln_post = nn.LayerNorm(self.width)
        self.proj_out = SNLinear(self.width, self.token_size, bias=True)

    def forward(self, x, token_counts):
        device = x[0].device
        grids = [[dim//ps for dim, ps in zip(vid.shape[1:], self.patch_size)] for vid in x]

        x = [rearrange(vid, 'c (nt pt) (nh ph) (nw pw) -> (nt nh nw) (c pt ph pw)', pt=self.patch_size[0], ph=self.patch_size[1], pw=self.patch_size[2]) for vid in x] # patching
        x = as_nested_tensor(x, layout=torch.jagged, device=device).contiguous() # BLC nested tensor
        x = self.proj_in(x) # returns BLC

        latent_tokens = [self.mask_token.repeat(num, self.width) for num in token_counts]
        x = as_nested_tensor([torch.cat([vid, tokens], dim=0) for vid, tokens in zip(x, latent_tokens)], layout=torch.jagged, device=device).contiguous()

        freqs = self.rope(x, grids, token_counts)
        x = self.model_layers(x, freqs)

        latent_tokens = as_nested_tensor([tokens[-num:] for tokens, num in zip(x, token_counts)], layout=torch.jagged, device=device).contiguous()

        latent_tokens = self.ln_post(latent_tokens)
        latent_tokens = self.proj_out(latent_tokens)
        
        return latent_tokens


class TiTokDecoder(nn.Module):
    def __init__(
            self,
            model_size="tiny",
            patch_size=(4, 8, 8),
            in_channels=5,
            out_channels=3,
        ):
        super().__init__()
        self.patch_size = patch_size
        self.token_size = in_channels
        self.out_channels = out_channels

        self.width, self.num_layers, self.num_heads, mlp_ratio = get_model_dims(model_size)
        scale = self.width ** -0.5

        self.rope = RoPE(self.width)

        self.mask_token = nn.Parameter(scale * torch.randn(1))

        self.proj_in = SNLinear(self.token_size, self.width, bias=True)
        self.ln_pre = nn.LayerNorm(self.width)

        self.model_layers = ResidualAttentionBlock(embed_dim=self.width, num_head=self.num_heads, mlp_ratio=mlp_ratio, num_layer=self.num_layers)

        self.proj_out = SNLinear(in_features=self.width, out_features=out_channels*math.prod(patch_size))


    def forward(self, x, out_grids):
        device = x.device
        token_counts = [sample.shape[0] for sample in x] # get L for every item in batch

        grids = [[dim//ps for dim, ps in zip(grid, self.patch_size)] for grid in out_grids]
        grid_sizes = [math.prod(grid) for grid in grids]

        x = self.proj_in(x)

        mask_tokens = [self.mask_token.repeat(grid_size, self.width) for grid_size in grid_sizes]        
        x = as_nested_tensor([torch.cat([masked_vid, tokens], dim=0) for masked_vid, tokens in zip(mask_tokens, x)], layout=torch.jagged, device=device).contiguous()

        x = self.ln_pre(x)

        freqs = self.rope(x, grids, token_counts)
        x = self.model_layers(x, freqs)

        x = as_nested_tensor([tokens[:grid_size] for tokens, grid_size in zip(x, grid_sizes)], layout=torch.jagged, device=device).contiguous() # keeps grads?
        x = self.proj_out(x)

        x = [rearrange(
                vid,
                '(nt nh nw) (c pt ph pw) -> c (nt pt) (nh ph) (nw pw)',
                nt=grid[0], nh=grid[1], nw=grid[2],
                pt=self.patch_size[0], ph=self.patch_size[1], pw=self.patch_size[2],
            )
            for vid, grid in zip(x, grids)]
        
        return x # list of video tensors in (CTHW) out