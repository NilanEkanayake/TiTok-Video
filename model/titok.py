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
import torch
import torch.nn as nn
from model.base.blocks import TiTokEncoder, TiTokDecoder
from model.quantizer.fsq import FSQ

class TiTok(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        titok_conf = config.model.titok
        token_size = len(titok_conf.fsq_levels)

        self.encoder = TiTokEncoder(
            model_size=titok_conf.encoder_size,
            patch_size=titok_conf.patch_size,
            in_channels=3,
            out_channels=token_size,
        )
        self.quantize = FSQ(levels=titok_conf.fsq_levels)
        self.decoder = TiTokDecoder(
            model_size=titok_conf.decoder_size,
            patch_size=titok_conf.patch_size,
            in_channels=token_size,
            out_channels=3,
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
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

    def encode(self, x, token_counts):
        x = self.encoder(x, token_counts)
        x_q, x_dict = self.quantize(x)
        x_dict['indices'] = torch.split(x_dict['indices'], token_counts, dim=0) # [B*L] -> [B, L]
        return x_q, x_dict
    
    def decode(self, x, token_counts, grids):
        x = self.decoder(x, token_counts, grids)
        return x
    
    def forward(self, x, token_counts):
        grids = [vid.shape[1:] for vid in x] # c|THW|
        x_q, out_dict = self.encode(x, token_counts)
        x = self.decode(x_q, token_counts, grids)
        return x, out_dict