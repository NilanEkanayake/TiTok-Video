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
from torchvision.transforms import v2, functional

def _expand_token(token, batch_size: int):
    return token.unsqueeze(0).expand(batch_size, -1, -1)

    
class TiTokEncoder(nn.Module):
    def __init__(self, model_config, dataset_config, token_size):
        super().__init__()
        self.model_config = model_config
        self.dataset_config = dataset_config

        self.spatial_size = dataset_config.resolution
        self.temporal_size = dataset_config.num_frames
        self.spatial_patch_size = model_config.spatial_patch_size
        self.temporal_patch_size = model_config.temporal_patch_size
        assert self.spatial_size % self.spatial_patch_size == 0 and self.temporal_size % self.temporal_patch_size == 0, "input dimensions should be evenly divisible by respective patch sizes"
        self.grid_size = ((self.spatial_size // self.spatial_patch_size) ** 2) * (self.temporal_size // self.temporal_patch_size)
        self.model_size = model_config.encoder_size
        self.num_latent_tokens = model_config.num_latent_tokens
        self.token_size = token_size
        self.in_channels = 3 # RGB

        self.width = {
                "tiny": 256,
                "small": 512,
                "base": 768,
                "large": 1024,
            }[self.model_size]
        self.num_layers = {
                "tiny": 4,
                "small": 8,
                "base": 12,
                "large": 24,
            }[self.model_size]
        self.num_heads = {
                "tiny": 4,
                "small": 8,
                "base": 12,
                "large": 16,
            }[self.model_size]
        
        st_dims = (self.temporal_patch_size, self.spatial_patch_size, self.spatial_patch_size)

        self.patch_embed = nn.Conv3d(
            in_channels=self.in_channels,
            out_channels=self.width,
            kernel_size=st_dims,
            stride=st_dims,
            padding=0,
            bias=True
        )
        
        scale = self.width ** -0.5

        self.latent_tokens = nn.Parameter(scale * torch.randn(self.num_latent_tokens, self.width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size, self.width))        

        self.model_layers = nn.ModuleList()
        
        #############################
        for i in range(self.num_layers):
            self.model_layers.append(ResidualAttentionBlock(d_model=self.width, n_head=self.num_heads, exp_res=model_config.exp_residual))
        #############################

        self.ln_post = nn.LayerNorm(self.width)

        self.linear_out = nn.Linear(self.width, self.token_size, bias=True)


    def forward(self, x):
        x = self.patch_embed(x)

        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = x + self.positional_embedding.to(x.dtype)

        latent_tokens = _expand_token(self.latent_tokens, x.shape[0]).to(x.dtype)

        x = torch.cat([x, latent_tokens], dim=1)

        for i in range(len(self.model_layers)):
            x = self.model_layers[i](x)

        latent_tokens = x[:, -self.num_latent_tokens:]
        latent_tokens = self.ln_post(latent_tokens)

        latent_tokens = self.linear_out(latent_tokens)
            
        return latent_tokens
    

class TiTokDecoder(nn.Module):
    def __init__(self, model_config, dataset_config, token_size):
        super().__init__()
        self.model_config = model_config
        self.dataset_config = dataset_config

        self.spatial_size = dataset_config.resolution
        self.temporal_size = dataset_config.num_frames
        self.spatial_patch_size = model_config.spatial_patch_size
        self.temporal_patch_size = model_config.temporal_patch_size
        self.grid_size = ((self.spatial_size // self.spatial_patch_size) ** 2) * (self.temporal_size // self.temporal_patch_size)

        self.out_channels = 3 # RGB
        self.model_size = model_config.decoder_size
        self.num_latent_tokens = model_config.num_latent_tokens
        self.token_size = token_size

        self.width = {
                "tiny": 256,
                "small": 512,
                "base": 768,
                "large": 1024,
            }[self.model_size]
        self.num_layers = {
                "tiny": 4,
                "small": 8,
                "base": 12,
                "large": 24,
            }[self.model_size]
        self.num_heads = {
                "tiny": 4,
                "small": 8,
                "base": 12,
                "large": 16,
            }[self.model_size]
        
        self.decoder_embed = nn.Linear(self.token_size, self.width, bias=True)
        
        scale = self.width ** -0.5

        self.mask_tokens = nn.Parameter(scale * torch.randn(self.grid_size, self.width)) # doesn't need pos emb

        self.latent_token_positional_embedding = nn.Parameter(scale * torch.randn(self.num_latent_tokens, self.width))
                
        self.ln_pre = nn.LayerNorm(self.width)

        self.model_layers = nn.ModuleList()

        #############################
        for i in range(self.num_layers):
            self.model_layers.append(ResidualAttentionBlock(d_model=self.width, n_head=self.num_heads, exp_res=model_config.exp_residual))
        #############################

        channel_depth = self.out_channels * self.temporal_patch_size * self.spatial_patch_size ** 2

        self.conv_out = nn.Sequential(
            nn.Conv3d(in_channels=self.width, out_channels=channel_depth, kernel_size=1, padding=0, bias=True),
            Rearrange('b (p1 p2 p3 c) t h w -> b c (t p1) (h p2) (w p3)',
                p1 = self.temporal_patch_size, p2 = self.spatial_patch_size, p3 = self.spatial_patch_size),
            nn.Conv3d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=True), # set padding to (kernel_size-1)//2
        )

        
    def forward(self, z_quantized):
        B, L, C = z_quantized.shape
        assert L == self.num_latent_tokens, f"{L}, {self.num_latent_tokens}"
        x = z_quantized

        x = self.decoder_embed(x)

        mask_tokens = _expand_token(self.mask_tokens, x.shape[0]).to(x.dtype)
        
        x = x + self.latent_token_positional_embedding
        x = torch.cat([mask_tokens, x], dim=1)
        
        x = self.ln_pre(x)

        for i in range(len(self.model_layers)):
            x = self.model_layers[i](x)

        x = x[:, :self.grid_size]
        
        x = x.permute(0, 2, 1).reshape(
            B,
            self.width,
            self.temporal_size // self.temporal_patch_size,
            self.spatial_size // self.spatial_patch_size,
            self.spatial_size // self.spatial_patch_size
        )

        x = self.conv_out(x.contiguous())
        return x