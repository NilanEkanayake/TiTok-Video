import torch
import torch.nn as nn
from causalvideovae.model.vae.modeling_wfvae import WFVAEModel
from diffusers import AutoencoderKLCogVideoX
from vidtok.modules.util import instantiate_from_config
from omegaconf import OmegaConf
import os

class VidTokVAEWrapper(nn.Module):
    def __init__(self, model_path="base_tokenizers/pretrained_models/vidtok_kl_causal_488_16chn.ckpt", embed_dim=16): # fix load issue
        super().__init__()
        
        self.embed_dim = embed_dim
        config_file = "base_tokenizers/vidtok/configs/" + os.path.basename(model_path).replace('.ckpt', '.yaml')
        config = OmegaConf.load(config_file)

        config.model.params.ckpt_path = model_path
        config.model.params.ignore_keys = []
        config.model.params.verbose = False
        self.vae = instantiate_from_config(config.model)

        for param in self.vae.parameters():
            param.requires_grad = False

        self.vae.eval()

    def encode(self, x):
        x = x.clamp(-1, 1)
        x = self.vae.encode(x)
        return x
    
    def decode(self, x):
        x = self.vae.decode(x)
        x = x.clamp(-1, 1)
        return x


class WFVAEWrapper(nn.Module): # works with 16-channel VAE from https://github.com/PKU-YuanGroup/WF-VAE and 8-channel VAE from https://github.com/PKU-YuanGroup/Open-Sora-Plan
    def __init__(self, model_path="base_tokenizers/pretrained_models/wfvae", embed_dim=16):
        super().__init__()

        self.embed_dim = embed_dim
        
        self.vae = WFVAEModel.from_pretrained(model_path)

        for param in self.vae.parameters():
            param.requires_grad = False

        self.vae.eval()

    def encode(self, x):
        x = x.clamp(-1, 1)
        x = self.vae.encode(x).latent_dist.sample()
        if self.embed_dim == 8:
            x = x * self.vae.config.scale[0]
        return x
    
    def decode(self, x):
        if self.embed_dim == 8:
            x = x / self.vae.config.scale[0]
        x = self.vae.decode(x).sample.contiguous()
        x = x.clamp(-1, 1)
        return x
    
    
class CogvideoXVAEWrapper(nn.Module): # [0, 1] rescaling needed for CogvideoX vae
    def __init__(self, model_path="base_tokenizers/pretrained_models/cogvideox", embed_dim=16):
        super().__init__()

        self.embed_dim = embed_dim

        self.vae = AutoencoderKLCogVideoX.from_pretrained(model_path)

        for param in self.vae.parameters():
            param.requires_grad = False

        self.vae.eval()

    def encode(self, x):
        x = x.clamp(-1, 1)
        x = (x + 1) / 2
        x = self.vae.encode(x)[0].sample()
        return x
    
    def decode(self, x):
        x = self.vae.decode(x).sample.contiguous()
        x = (x * 2) - 1
        x = x.clamp(-1, 1)
        return x