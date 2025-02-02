import torch
from torch import nn
from model.base.transformer_block import ResidualAttentionBlock
from einops.layers.torch import Rearrange

def _expand_token(token, batch_size: int):
    return token.unsqueeze(0).expand(batch_size, -1, -1)

class TiTokDiscriminator(nn.Module):
    def __init__(
            self,
            num_layers=2,
            d_model=128,
            num_heads=2,
            spatial_patch_size=4,
            in_channels=8,
            in_spatial_size=24,
        ):
        super().__init__()
        self.in_channels = in_channels
        self.spatial_size = in_spatial_size
        self.spatial_patch_size = spatial_patch_size

        assert self.spatial_size % self.spatial_patch_size == 0, "input dimensions should be evenly divisible by respective patch sizes"

        self.grid_size = ((self.spatial_size // self.spatial_patch_size) ** 2)

        self.width = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads

        in_channel_depth = self.in_channels * self.spatial_patch_size ** 2
        scale = self.width ** -0.5

        self.latent_tokens = nn.Parameter(scale * torch.randn(1, self.width)) 
        self.latent_token_positional_embedding = nn.Parameter(scale * torch.randn(1, self.width))

        self.conv_in = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, padding=1, bias=True),
            Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w',
                p1 = self.spatial_patch_size, p2 = self.spatial_patch_size),
            nn.Conv2d(in_channels=in_channel_depth, out_channels=self.width, kernel_size=1, padding=0, bias=True),
            )
        
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size, self.width))

        self.ln_pre = nn.LayerNorm(self.width)

        self.model_layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.model_layers.append(ResidualAttentionBlock(d_model=self.width, n_head=self.num_heads, exp_res=False))

        self.ln_post = nn.LayerNorm(self.width)

        self.linear_out = nn.Linear(self.width, 1, bias=True)


    def forward(self, x):
        x = self.conv_in(x)

        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = x + self.positional_embedding.to(x.dtype)

        latent_tokens = _expand_token(self.latent_tokens, x.shape[0]).to(x.dtype)
        latent_tokens = latent_tokens + self.latent_token_positional_embedding.to(x.dtype)
        x = torch.cat([x, latent_tokens], dim=1)

        x = self.ln_pre(x)

        for i in range(len(self.model_layers)):
            x = self.model_layers[i](x)

        x = x[:, -1:]

        x = self.ln_post(x)

        x = self.linear_out(x)

        return x