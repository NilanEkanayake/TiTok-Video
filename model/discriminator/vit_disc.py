import torch
from torch import nn
from model.base.transformer import ResidualAttentionBlock, GEGLU
from model.base.blocks import get_model_dims
from einops.layers.torch import Rearrange
import math

def _expand_token(token, batch_size: int):
    return token.unsqueeze(0).expand(batch_size, -1, -1)

class ViTDiscriminator(nn.Module):
    def __init__(
            self,
            model_size="tiny",
            in_grid=(8, 128, 128), # THW
            in_channels=3,
            patch_size=(4, 8, 8),
            out_tokens=1,
        ):
        super().__init__()
        self.num_latent_tokens = out_tokens

        self.grid = [x//y for x, y in zip(in_grid, patch_size)]
        self.grid_size = math.prod(self.grid)
        self.in_channels = in_channels

        self.width, self.num_layers, self.num_heads, mlp_ratio = get_model_dims(model_size)
        scale = self.width ** -0.5

        self.latent_tokens = nn.Parameter(scale * torch.randn(self.num_latent_tokens, self.width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size, self.width))

        self.proj_in = nn.Sequential(
            Rearrange('b c (nt pt) (nh ph) (nw pw) -> b (nt nh nw) (c pt ph pw)', pt=patch_size[0], ph=patch_size[1], pw=patch_size[2]),
            nn.Linear(in_features=in_channels*math.prod(patch_size), out_features=self.width),
        )

        self.model_layers = ResidualAttentionBlock(embed_dim=self.width, num_head=self.num_heads, mlp_ratio=mlp_ratio, num_layer=self.num_layers)

        inner_dim = int(mlp_ratio * (2 / 3) * self.width)
        inner_dim = 32 * ((inner_dim + 32 - 1) // 32)
        self.proj_out =  nn.Sequential(
            nn.LayerNorm(self.width),
            nn.Linear(self.width, inner_dim * 2, bias=False),
            GEGLU(),
            nn.Linear(inner_dim, 1, bias=False), # 1 channel out
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

    def forward(self, x):
        x = self.proj_in(x)
        
        x = x + self.positional_embedding.to(x.dtype)
        latent_tokens = _expand_token(self.latent_tokens, x.shape[0]).to(x.dtype)

        x = torch.cat([x, latent_tokens], dim=1).contiguous()

        x = self.model_layers(x)

        out_tokens = x[:, -self.num_latent_tokens:]
        out_tokens = self.proj_out(out_tokens)

        return out_tokens # [B, L, 1]