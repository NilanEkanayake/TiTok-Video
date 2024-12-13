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
import json
from omegaconf import OmegaConf
from pathlib import Path
from model.base.distillation_modules import TiTokEncoder, TiTokDecoder
from model.base.base_model import BaseModel
from model.quantizer.vae import SampleVAE
from model.quantizer.fsq import FSQ
from model.quantizer.bsq import BSQQuantizer
from huggingface_hub import PyTorchModelHubMixin

class TiTok(BaseModel, PyTorchModelHubMixin, tags=["arxiv:2304.12244", "video-tokenization"], license="mit"):
    def __init__(self, config):

        if isinstance(config, dict):
            config = OmegaConf.create(config)

        super().__init__()
        self.config = config

        self.quant_mode = config.model.titok.quant_mode

        if self.quant_mode == "fsq":
            self.quantize = FSQ(config.model.titok.fsq_levels)
            self.token_size = len(config.model.titok.fsq_levels)
        elif self.quant_mode == "vae":
            self.quantize = SampleVAE()
            self.token_size = config.model.titok.token_size
        elif self.quant_mode == "bsq":
            self.quantize = BSQQuantizer(config)
            self.token_size = config.model.titok.token_size
        else:
            raise Exception(f"Unknown quant mode: {self.quant_mode}")
        
        self.encoder = TiTokEncoder(config, self.token_size)
        self.decoder = TiTokDecoder(config, self.token_size)

        self.num_latent_tokens = config.model.titok.num_latent_tokens
        scale = self.encoder.width ** -0.5
        self.latent_tokens = nn.Parameter(scale * torch.randn(self.num_latent_tokens, self.encoder.width))
        
        self.apply(self._init_weights)
        
    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights and config to a local directory."""
        # Assume 'self.config' is your DictConfig object
        # Convert to a regular dictionary
        dict_config = OmegaConf.to_container(self.config)
        # Save as JSON
        file_path = Path(save_directory) / "config.json"
        with open(file_path, 'w') as json_file:
            json.dump(dict_config, json_file, indent=4)
        super()._save_pretrained(save_directory)

    def _init_weights(self, module):
        """ Initialize the weights.
            :param:
                module -> torch.nn.Module: module to initialize
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode(self, x):
        z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens)
        z = self.quantize(z.contiguous())
        return z
    
    def decode(self, z):
        decoded = self.decoder(z)
        return decoded
    
    def forward(self, x):
        z, result_dict = self.encode(x)
        decoded = self.decode(z)
        return decoded, result_dict





    