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
from model.base.blocks import TiTokEncoder, TiTokDecoder, init_weights
from model.quantizer.fsq import FSQ

class TiTok(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        titok_conf = config.tokenizer.model
        token_size = len(titok_conf.fsq_levels)

        self.encoder = TiTokEncoder(
            model_size=titok_conf.encoder_size,
            patch_size=titok_conf.patch_size,
            in_channels=3,
            out_channels=token_size,
            max_grid=self.config.training.sampling.max_grid,
            max_tokens=self.config.training.sampling.num_token_range[1],
        )
        self.quantize = FSQ(levels=titok_conf.fsq_levels)
        self.decoder = TiTokDecoder(
            model_size=titok_conf.decoder_size,
            patch_size=titok_conf.patch_size,
            in_channels=token_size,
            out_channels=3,
            max_grid=self.config.training.sampling.max_grid,
            max_tokens=self.config.training.sampling.num_token_range[1],
        )

        self.apply(init_weights)

    def encode(self, x, token_counts):
        x = self.encoder(x, token_counts)
        x_q, x_dict = self.quantize(x)
        x_dict['indices'] = torch.split(x_dict['indices'], token_counts, dim=0) # [B*L] -> [B, L]
        return x_q, x_dict
    
    def decode(self, x, token_counts, grids):
        x = self.decoder(x, token_counts, grids)
        return x
    
    def decode_indices(self, indices, grids):
        token_counts = [x.shape[0] for x in indices]
        x_q = self.quantize.indices_to_codes(indices).to(indices.device, next(self.decoder.parameters()).dtype) # expects B*L in
        return self.decode(x_q, token_counts, grids)
    
    def forward(self, x, token_counts):
        grids = [vid.shape[1:] for vid in x] # c|THW|
        x_q, out_dict = self.encode(x, token_counts)
        x = self.decode(x_q, token_counts, grids)
        return x, out_dict