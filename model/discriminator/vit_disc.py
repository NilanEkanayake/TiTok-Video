import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base.transformer import ResidualAttentionBlock, GEGLU
from model.base.sigma_reparam import SNLinear
from model.base.rope import RoPE

from torch.nested import nested_tensor, as_nested_tensor
from model.base.blocks import get_model_dims

from einops.layers.torch import Rearrange
from einops import rearrange
import math


class ViTDiscriminator(nn.Module):
    def __init__(
            self,
            model_size="tiny",
            patch_size=(4, 8, 8),
            out_tokens=1,
            in_channels=3,
            out_channels=1,
        ):
        super().__init__()

        self.patch_size = patch_size
        self.num_out_tokens = out_tokens

        self.width, self.num_layers, self.num_heads, mlp_ratio = get_model_dims(model_size)
        scale = self.width ** -0.5

        self.rope = RoPE(self.width)

        self.mask_token = nn.Parameter(scale * torch.randn(1))
        self.proj_in = SNLinear(in_features=in_channels*math.prod(patch_size), out_features=self.width)
        self.model_layers = ResidualAttentionBlock(embed_dim=self.width, num_head=self.num_heads, mlp_ratio=mlp_ratio, num_layer=self.num_layers)

        inner_dim = int(mlp_ratio * (2 / 3) * self.width)
        inner_dim = 32 * ((inner_dim + 32 - 1) // 32)
        self.proj_out =  nn.Sequential(
            nn.LayerNorm(self.width),
            SNLinear(self.width, inner_dim * 2, bias=False),
            GEGLU(),
            SNLinear(inner_dim, out_channels, bias=False), # 1 channel out
        )

        # just use single linear as out proj?

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


    def forward(self, x): # x is list of CTHW vids
        device = x[0].device
        grids = [[dim//ps for dim, ps in zip(vid.shape[1:], self.patch_size)] for vid in x] # c|THW|

        x = [rearrange(vid, 'c (nt pt) (nh ph) (nw pw) -> (nt nh nw) (c pt ph pw)', pt=self.patch_size[0], ph=self.patch_size[1], pw=self.patch_size[2]) for vid in x] # patching
        x = as_nested_tensor(x, layout=torch.jagged, device=device).contiguous() # BLC nested tensor
        x = self.proj_in(x) # returns BLC

        latent_tokens = self.mask_token.repeat(self.num_out_tokens, self.width) # LC - use absolute pos emb? not worth effort?
        x = as_nested_tensor([torch.cat([vid, latent_tokens], dim=0) for vid in x], layout=torch.jagged, device=device).contiguous()

        freqs = self.rope(x, grids, [self.num_out_tokens]*len(grids))
        x = self.model_layers(x, freqs)

        out_tokens = torch.stack([tokens[-self.num_out_tokens:] for tokens in x], dim=0).contiguous() # fixed number of output tokens, can use dense tensor

        out_tokens = self.proj_out(out_tokens)
        return out_tokens # tensor out = [B, out_tokens, out_channels]