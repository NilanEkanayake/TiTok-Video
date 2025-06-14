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
from model.base.transformer import ResidualAttentionBlock

from einops.layers.torch import Rearrange
import math

def _expand_token(token, batch_size: int):
    return token.unsqueeze(0).expand(batch_size, -1, -1)

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
            in_grid=(4, 16, 16), # THW
            in_channels=512,
            out_tokens=512,
            out_channels=5,
        ):
        super().__init__()

        self.num_latent_tokens = out_tokens
        self.token_size = out_channels

        self.grid = in_grid
        self.grid_size = math.prod(self.grid)
        self.in_channels = in_channels

        self.width, self.num_layers, self.num_heads, mlp_ratio = get_model_dims(model_size)
        scale = self.width ** -0.5

        self.latent_tokens = nn.Parameter(scale * torch.randn(self.num_latent_tokens, self.width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size, self.width))

        self.proj_in = nn.Linear(self.in_channels, self.width, bias=True)
        
        #############################
        self.model_layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.model_layers.append(ResidualAttentionBlock(d_model=self.width, n_head=self.num_heads, mlp_ratio=mlp_ratio))
        #############################

        self.ln_post = nn.LayerNorm(self.width)
        self.proj_out = nn.Linear(self.width, self.token_size, bias=True)


    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], -1) # BCTHW -> BC(THW)/BCL
        x = x.permute(0, 2, 1) # BCL -> BLC
        x = self.proj_in(x)

        x = x + self.positional_embedding.to(x.dtype)
        latent_tokens = _expand_token(self.latent_tokens, x.shape[0]).to(x.dtype)

        x = torch.cat([x, latent_tokens], dim=1).contiguous()

        for i in range(len(self.model_layers)):
            x = self.model_layers[i](x)

        latent_tokens = x[:, -self.num_latent_tokens:]
        latent_tokens = self.ln_post(latent_tokens)
        latent_tokens = self.proj_out(latent_tokens)
        
        return latent_tokens


class TiTokDecoder(nn.Module):
    def __init__(
            self,
            model_size="tiny",
            in_tokens=512,
            in_channels=5,
            out_grid=(4, 16, 16), # THW
            out_channels=512,
        ):
        super().__init__()

        self.num_latent_tokens = in_tokens
        self.token_size = in_channels

        self.grid = out_grid
        self.grid_size = math.prod(self.grid)
        self.out_channels = out_channels
        

        self.width, self.num_layers, self.num_heads, mlp_ratio = get_model_dims(model_size)
        scale = self.width ** -0.5

        self.mask_tokens = nn.Parameter(scale * torch.randn(self.grid_size, self.width)) # doesn't need pos emb
        self.latent_token_positional_embedding = nn.Parameter(scale * torch.randn(self.num_latent_tokens, self.width))

        self.proj_in = nn.Linear(self.token_size, self.width, bias=True)
        self.ln_pre = nn.LayerNorm(self.width)

        #############################
        self.model_layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.model_layers.append(ResidualAttentionBlock(d_model=self.width, n_head=self.num_heads, mlp_ratio=mlp_ratio))
        #############################

        self.proj_out = nn.Linear(self.width, self.out_channels, bias=True)

    def forward(self, x):
        x = self.proj_in(x)
        mask_tokens = _expand_token(self.mask_tokens, x.shape[0]).to(x.dtype)
        
        x = x + self.latent_token_positional_embedding
        x = torch.cat([mask_tokens, x], dim=1).contiguous()
        x = self.ln_pre(x)

        for i in range(len(self.model_layers)):
            x = self.model_layers[i](x)

        x = x[:, :self.grid_size]
        x = self.proj_out(x)
        x = x.permute(0, 2, 1).reshape([x.shape[0], self.out_channels] + self.grid) # BLC -> BCL -> BCTHW
        return x