"""This file contains the model definition of TiTok.

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
"""
import torch.nn as nn
from model.base.blocks import TiTokEncoder, TiTokDecoder
from model.base.leanVAE import ResNAFEncoder, ResNAFDecoder
from model.base.patcher_utils import Patcher, UnPatcher # DWT
from model.quantizer.fsq import FSQ

class TiTok(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        resnaf_conf = config.model.resnaf
        titok_conf = config.model.titok

        self.grid = [ # THW
            config.dataset.num_frames,
            config.dataset.resolution,
            config.dataset.resolution,
        ]

        self.patch_size = [
            resnaf_conf.temporal_patch_size,
            resnaf_conf.spatial_patch_size,
            resnaf_conf.spatial_patch_size,
        ]

        assert all(x % 2 == 0 for x in self.patch_size), "patch sizes must be multiple of two"
        assert all(x % y == 0 for x, y in zip(self.grid, self.patch_size)), "input dimensions should be evenly divisible by respective patch sizes"

        sep_layers = resnaf_conf.sep_layers
        fusion_layers = resnaf_conf.fusion_layers
        l_dim = resnaf_conf.low_dims
        h_dim = resnaf_conf.high_dims

        token_size = len(titok_conf.fsq_levels)
        titok_grid = [x//y for x, y in zip(self.grid, self.patch_size)]

        self.dwt = Patcher()
        self.pre_encoder = ResNAFEncoder(
            sep_layers,
            fusion_layers,
            [x//2 for x in self.patch_size], # initial /2 already done by DWT.
            l_dim,
            h_dim
        )
        self.encoder = TiTokEncoder(
            model_size=titok_conf.encoder_size,
            in_grid=titok_grid,
            in_channels=l_dim+h_dim,
            out_tokens=titok_conf.num_latent_tokens,
            out_channels=token_size,
        )
        self.quantize = FSQ(levels=titok_conf.fsq_levels)
        self.decoder = TiTokDecoder(
            model_size=titok_conf.decoder_size,
            in_tokens=titok_conf.num_latent_tokens,
            in_channels=token_size,
            out_grid=titok_grid,
            out_channels=l_dim+h_dim,
        )
        self.post_decoder = ResNAFDecoder(
            sep_layers,
            fusion_layers,
            [x//2 for x in self.patch_size],
            l_dim,
            h_dim
        )
        self.idwt = UnPatcher()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
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


    def encode(self, x):
        x = x/2 # [-1, 1] -> [-0.5, 0.5] needed?
        x_dwt = self.dwt(x)
        x = self.pre_encoder(x_dwt)
        x = self.encoder(x)
        x_q, x_dict = self.quantize(x)
        return x_q, x_dwt, x_dict
    
    def decode(self, x):
        x = self.decoder(x)
        x_dwt = self.post_decoder(x)
        x = self.idwt(x_dwt)
        x = x*2 # needed?
        return x, x_dwt
    
    def forward(self, x):
        x, target_dwt, out_dict = self.encode(x)        
        x, recon_dwt = self.decode(x)

        out_dict['target_dwt'] = target_dwt
        out_dict['recon_dwt'] = recon_dwt

        return x, out_dict