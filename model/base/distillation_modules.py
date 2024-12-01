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
from model.base.transformer_block import ResidualAttentionBlock, _expand_token
    
class TiTokEncoder(nn.Module):
    def __init__(self, config, token_size):
        super().__init__()
        self.config = config
        # 4x8x8
        self.spatial_size = config.video_dataset.params.resolution // config.model.vae_model.spatial_compression 
        self.temporal_size = config.video_dataset.params.num_frames // config.model.vae_model.temporal_compression 
        self.spatial_patch_size = config.model.vq_model.vit_enc_spatial_patch_size
        self.temporal_patch_size = config.model.vq_model.vit_enc_temporal_patch_size
        self.grid_size = ((self.spatial_size // self.spatial_patch_size) ** 2) * (self.temporal_size // self.temporal_patch_size)
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        self.token_size = token_size
        self.model_size = config.model.vq_model.vit_enc_model_size

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
        
        
        self.vae_conv_size = (self.temporal_patch_size, self.spatial_patch_size, self.spatial_patch_size)

        self.patch_embed = nn.Conv3d(
            in_channels=config.model.vae_model.latent_channels, 
            out_channels=self.width, 
            kernel_size=self.vae_conv_size,
            stride=self.vae_conv_size,
            # padding=(1, 0, 0),
            bias=True
        )

        scale = self.width ** -0.5

        self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size, self.width))
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.width))
        
        self.ln_pre = nn.LayerNorm(self.width)

        self.width_mid = self.width

        self.width_out = self.width

        self.model_layers = nn.ModuleList()
        
        #############################
        for i in range(self.num_layers):
            self.model_layers.append(ResidualAttentionBlock(d_model=self.width_out, n_head=self.num_heads))
        #############################

        self.ln_post = nn.LayerNorm(self.width_out)

        self.linear_out = nn.Linear(self.width_out, self.token_size, bias=True)

    def forward(self, x, latent_tokens):
        x = self.patch_embed(x)

        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1) # shape = [*, grid, width]

        x = x + self.positional_embedding.to(x.dtype)
        

        latent_tokens = _expand_token(latent_tokens, x.shape[0]).to(x.dtype)
        latent_tokens = latent_tokens + self.latent_token_positional_embedding.to(x.dtype)
        x = torch.cat([x, latent_tokens], dim=1)

        x = self.ln_pre(x)

        for i in range(len(self.model_layers)):
            x = self.model_layers[i](x)

        latent_tokens = x[:, self.grid_size:]
        latent_tokens = self.ln_post(latent_tokens)

        latent_tokens = self.linear_out(latent_tokens)
            
        return latent_tokens
    

class TiTokDecoder(nn.Module):
    def __init__(self, config, token_size):
        super().__init__()
        self.config = config

        self.spatial_size = config.video_dataset.params.resolution // config.model.vae_model.spatial_compression 
        self.temporal_size = config.video_dataset.params.num_frames // config.model.vae_model.temporal_compression 
        self.spatial_patch_size = config.model.vq_model.vit_enc_spatial_patch_size
        self.temporal_patch_size = config.model.vq_model.vit_enc_temporal_patch_size
        self.grid_size = ((self.spatial_size // self.spatial_patch_size) ** 2) * (self.temporal_size // self.temporal_patch_size)

        self.model_size = config.model.vq_model.vit_dec_model_size
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        self.token_size = token_size
        self.model_size = config.model.vq_model.vit_dec_model_size

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

        self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size, self.width))
        
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1, self.width))
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.width))
        
        self.ln_pre = nn.LayerNorm(self.width)

        self.width_mid = self.width
        self.width_out = self.width

        self.model_layers = nn.ModuleList()
        
        #############################
        for i in range(self.num_layers):
            self.model_layers.append(ResidualAttentionBlock(d_model=self.width_out, n_head=self.num_heads))
        #############################

        self.ln_post = nn.LayerNorm(self.width_out)

        self.vae_conv_size = (self.temporal_patch_size, self.spatial_patch_size, self.spatial_patch_size)
        self.conv_out = nn.ConvTranspose3d(
            in_channels=self.width_out, 
            out_channels=config.model.vae_model.latent_channels, 
            kernel_size=self.vae_conv_size, 
            stride=self.vae_conv_size, 
            # output_padding=(1, 0, 0), 
            bias=True
        )
        
        
    def forward(self, z_quantized):
        N, W, C = z_quantized.shape
        assert W == self.num_latent_tokens, f"{W}, {self.num_latent_tokens}"
        x = z_quantized

        x = self.decoder_embed(x)

        batchsize, seq_len, _ = x.shape

        mask_tokens = self.mask_token.repeat(batchsize, self.grid_size, 1).to(x.dtype)

        mask_tokens = mask_tokens + self.positional_embedding.to(mask_tokens.dtype)
        x = x + self.latent_token_positional_embedding[:seq_len]
        x = torch.cat([mask_tokens, x], dim=1)
        
        x = self.ln_pre(x)

        for i in range(len(self.model_layers)):
            x = self.model_layers[i](x)

        x = x[:, :self.grid_size]

        x = self.ln_post(x)
        
        # N L D -> N D H W
        x = x.permute(0, 2, 1).reshape(
            batchsize,
            self.width_out,
            self.temporal_size // self.temporal_patch_size,
            self.spatial_size // self.spatial_patch_size, 
            self.spatial_size // self.spatial_patch_size
        )

        x = self.conv_out(x.contiguous())
        
        return x