import torch
from torch import nn
from model.base.transformer import ResidualAttentionBlock
from einops.layers.torch import Rearrange

def _expand_token(token, batch_size: int):
    return token.unsqueeze(0).expand(batch_size, -1, -1)

class TiTokDiscriminator(nn.Module):
    def __init__(
            self,
            num_layers=4, # 4
            num_heads=4, # 4
            d_model=256, # 256
            in_channels=3,
            in_spatial_size=128,
            in_temporal_size=8,
            spatial_patch_size=8, # 16
            temporal_patch_size=4,
        ):
        super().__init__()
        self.in_channels = in_channels
        self.spatial_size = in_spatial_size
        self.temporal_size = in_temporal_size
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size

        assert self.spatial_size % self.spatial_patch_size == 0, "input dimensions should be evenly divisible by respective patch sizes"

        self.grid_size = ((self.spatial_size // self.spatial_patch_size) ** 2) * (self.temporal_size // self.temporal_patch_size)

        self.width = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads

        in_channel_depth = self.in_channels * self.temporal_patch_size * self.spatial_patch_size ** 2
        scale = self.width ** -0.5

        self.latent_token = nn.Parameter(scale * torch.randn(1, self.width)) 

        self.conv_in = nn.Sequential(
            Rearrange('b c (t p1) (h p2) (w p3) -> b (p1 p2 p3 c) t h w',
                p1 = self.temporal_patch_size, p2 = self.spatial_patch_size, p3 = self.spatial_patch_size),
            nn.Conv3d(in_channels=in_channel_depth, out_channels=self.width, kernel_size=1, padding=0, bias=True),
        )
        
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size, self.width))

        self.model_layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.model_layers.append(ResidualAttentionBlock(d_model=self.width, n_head=self.num_heads, exp_res=False))

        self.ln_post = nn.LayerNorm(self.width)

        self.linear_out = nn.Sequential(
            nn.Linear(self.width, self.width * 4),
            nn.GELU(),
            nn.Linear(self.width * 4, 1),
        )

        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """ Initialize the weights.
            :param:
                module -> torch.nn.Module: module to initialize
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, z):
        z = self.conv_in(z)
        
        z = z.reshape(z.shape[0], z.shape[1], -1)
        z = z.permute(0, 2, 1)
        z = z + self.positional_embedding.to(z.dtype)
        
        latent_token = _expand_token(self.latent_token, z.shape[0]).to(z.dtype)
        z = torch.cat([z, latent_token], dim=1)
        
        for i in range(len(self.model_layers)):
            z = self.model_layers[i](z)
        
        z = self.ln_post(z)
        z = self.linear_out(z)
        z = z[:, -1:]
        
        return z