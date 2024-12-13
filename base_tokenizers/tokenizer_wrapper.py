import torch
import torch.nn as nn
from causalvideovae.model.vae.modeling_wfvae import WFVAEModel
from diffusers import AutoencoderKLCogVideoX

class WFVAEWrapper(nn.Module): # works with 16-ch vae from https://github.com/PKU-YuanGroup/WF-VAE
    def __init__(self, model_path="base_tokenizers/pretrained_models/wfvae", dtype=torch.bfloat16, embed_dim=16):
        super().__init__()

        self.embed_dim = embed_dim
        self.torch_dtype=dtype

        self.vae = WFVAEModel.from_pretrained(model_path, torch_dtype=dtype)
        self.vae.eval()

    def encode(self, x):
        with torch.no_grad():
            orig_dtype = x.dtype
            # input is in range [0, 1], vae needs [-1, 1]
            x = (x*2)-1
            x = x.clamp(-1, 1)
            x = self.vae.encode(x.to(self.torch_dtype)).latent_dist.sample()
        return x.to(orig_dtype)
    
    def decode(self, x):
        with torch.no_grad():
            orig_dtype = x.dtype
            x = self.vae.decode(x.to(self.torch_dtype)).sample.contiguous()
            x = x.clamp(-1, 1)
            x = (x+1)/2 # rescale to [0, 1]
        return x.to(orig_dtype)

    def dtype(self):
        return self.torch_dtype
    
class CogvideoXVAEWrapper(nn.Module): # rescaling not needed for CogvideoX vae
    def __init__(self, model_path="base_tokenizers/pretrained_models/cogvideox", dtype=torch.bfloat16, embed_dim=16):
        super().__init__()

        self.embed_dim = embed_dim
        self.torch_dtype=dtype

        self.vae = AutoencoderKLCogVideoX.from_pretrained(model_path, torch_dtype=dtype)
        self.vae.eval()

    def encode(self, x):
        with torch.no_grad():
            orig_dtype = x.dtype
            x = x.clamp(0, 1)
            x = self.vae.encode(x.to(self.torch_dtype))[0].sample()
        return x.to(orig_dtype)
    
    def decode(self, x):
        with torch.no_grad():
            orig_dtype = x.dtype
            x = self.vae.decode(x.to(self.torch_dtype)).sample.contiguous()
            x = x.clamp(0, 1)
        return x.to(orig_dtype)

    def dtype(self):
        return self.torch_dtype
