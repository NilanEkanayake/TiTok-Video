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
import json
from omegaconf import OmegaConf
from pathlib import Path
from model.base.distillation_modules import TiTokEncoder, TiTokDecoder
from model.quantizer.fsq import FSQ
from huggingface_hub import PyTorchModelHubMixin

class TiTok(nn.Module, PyTorchModelHubMixin, tags=["arxiv:2304.12244", "video-tokenization"], license="mit"):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.quantize = FSQ(levels=config.model.titok.fsq_levels)
        token_size = len(config.model.titok.fsq_levels)
        
        self.encoder = TiTokEncoder(config.model.titok, config.model.vae, config.dataset, token_size)
        self.decoder = TiTokDecoder(config.model.titok, config.model.vae, config.dataset, token_size)
                
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
        z = self.encoder(pixel_values=x)
        z = self.quantize(z.contiguous())
        return z
    
    def decode(self, z):
        decoded = self.decoder(z)
        return decoded
    
    def forward(self, x):
        z, result_dict = self.encode(x)
        decoded = self.decode(z)
        return decoded, result_dict  