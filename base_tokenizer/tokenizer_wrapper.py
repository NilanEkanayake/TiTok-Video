import torch
import torch.nn as nn
from causalvideovae.model.vae.modeling_wfvae import WFVAEModel

class WFVAEWrapper(nn.Module):
    def __init__(self, model_path="base_tokenizers/pretrained_model", dtype=torch.bfloat16, id="wfvae-L16", embed_dim=16):
        super().__init__()

        self.embed_dim = embed_dim
        self.torch_dtype=dtype

        self.vae = WFVAEModel.from_pretrained(model_path, torch_dtype=dtype)
        self.vae.eval()

    def encode(self, x):
        with torch.no_grad():   
            x = self.vae.encode(x.to(self.torch_dtype)).latent_dist.sample().to(x.dtype)
            return x
    
    def decode(self, x):
        with torch.no_grad():   
            x = self.vae.decode(x.to(self.torch_dtype)).sample.contiguous().to(x.dtype)
        return x

    def dtype(self):
        return self.torch_dtype

