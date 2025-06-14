import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math

from einops.layers.torch import Rearrange
from einops import rearrange
import numpy as np

class PEG3D(nn.Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()
        self.ds_conv = nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, groups=dim)
    
    def forward(self, x):
        x = self.ds_conv(x.contiguous())
        return x


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def ffd(dim, mult=4, dropout=0.):
    inner_dim = int(mult * (2 / 3) * dim)
    return nn.Sequential(
        Rearrange('b c t h w -> b t h w c'),
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias=False),
        Rearrange('b t h w c -> b c t h w'),
    )

class NAF(nn.Module):
    def __init__(self,
                 num_layer, 
                 dim,
                 ): 
        super(NAF, self).__init__()
        self.num_layer = num_layer
        self.dconv_layer = nn.Sequential()
        self.ffd_layer = nn.Sequential()
        for _ in range(num_layer):
            self.ffd_layer.append(ffd(dim, 4)) 
            self.dconv_layer.append(PEG3D(dim))
   
    def forward(self, x):
        for i in range(self.num_layer):
            x = self.dconv_layer[i](x)
            x = self.ffd_layer[i](x) 
        return x


class ResNAF(nn.Module):
    def __init__(self,
                 num_layer, 
                 dim,
                 ): 
        super(ResNAF, self).__init__()
        self.num_layer = num_layer
        self.dconv_layer = nn.Sequential()
        self.ffd_layer = nn.Sequential()
        for _ in range(num_layer):
            self.ffd_layer.append(ffd(dim, 4)) 
            self.dconv_layer.append(PEG3D(dim))
   
    def forward(self, x):
        for i in range(self.num_layer):
            x = x + self.dconv_layer[i](x)
            x = x + self.ffd_layer[i](x) 
        return x
    
class DualFusionDownBlock(nn.Module):
    def __init__(
            self,
            in_channels_low=3,
            in_channels_high=21,
            out_channels_low=128,
            out_channels_high=384,
            patch_size=(2, 4, 4),
        ):
        super().__init__()

        self.in_channels_low = in_channels_low
        t, h, w = patch_size

        self.linear_low = nn.Sequential(
            Rearrange('b c (nt pt) (nh ph) (nw pw) -> b nt nh nw (c pt ph pw)', pt=t, ph=h, pw=w),
            nn.Linear(in_features=in_channels_low*t*h*w, out_features=out_channels_low),
            Rearrange('b t h w c -> b c t h w'),
        )

        self.linear_high = nn.Sequential(
            Rearrange('b c (nt pt) (nh ph) (nw pw) -> b nt nh nw (c pt ph pw)', pt=t, ph=h, pw=w),
            nn.Linear(in_features=in_channels_high*t*h*w, out_features=out_channels_high),
            Rearrange('b t h w c -> b c t h w'),
        )


    def forward(self, x):
        x_low = x[:, :self.in_channels_low]
        x_high = x[:, self.in_channels_low:]

        x_low = self.linear_low(x_low)
        x_high = self.linear_high(x_high)
        return x_low, x_high
    
    
class DualFusionUpBlock(nn.Module):
    def __init__(
            self,
            in_channels_low=128,
            in_channels_high=384,
            out_channels_low=3,
            out_channels_high=21,
            patch_size=(2, 4, 4),
        ):
        super().__init__()

        self.in_channels_low = in_channels_low
        t, h, w = patch_size

        self.linear_low = nn.Sequential(
            Rearrange('b c t h w -> b t h w c'),
            nn.Linear(in_features=in_channels_low, out_features=out_channels_low*t*h*w),
            Rearrange('b nt nh nw (c pt ph pw) -> b c (nt pt) (nh ph) (nw pw)', pt=t, ph=h, pw=w),
        )

        self.linear_high = nn.Sequential(
            Rearrange('b c t h w -> b t h w c'),
            nn.Linear(in_features=in_channels_high, out_features=out_channels_high*t*h*w),
            Rearrange('b nt nh nw (c pt ph pw) -> b c (nt pt) (nh ph) (nw pw)', pt=t, ph=h, pw=w),
        )

    def forward(self, x_low, x_high):
        x_low = self.linear_low(x_low)
        x_high = self.linear_high(x_high)

        x = torch.cat([x_low, x_high], dim=1)
        return x

    
class ResNAFEncoder(nn.Module):
    def __init__(
        self,
        sep_num_layer=2,
        fusion_num_layer=4,
        patch_size=(2, 4, 4),
        l_dim=128,
        h_dim=384,
        ):
        super().__init__() 

        self.patch_embed = DualFusionDownBlock(
            in_channels_low=3,
            in_channels_high=21,
            out_channels_low=l_dim,
            out_channels_high=h_dim,
            patch_size=patch_size
        )

        self.low_layer = ResNAF(num_layer=sep_num_layer, dim=l_dim)
        self.high_layer = ResNAF(num_layer=sep_num_layer, dim=h_dim)
        self.fusion_layer = ResNAF(num_layer=fusion_num_layer, dim=l_dim+h_dim)

    def forward(self, x):
        x_low, x_high = self.patch_embed(x)
        x_low = self.low_layer(x_low.contiguous())
        x_high = self.high_layer(x_high.contiguous())

        x = torch.cat([x_low, x_high], dim=1).contiguous() # channel concat
        x = self.fusion_layer(x)
        return x
    
class ResNAFDecoder(nn.Module):
    def __init__(
        self,
        sep_num_layer=2,
        fusion_num_layer=4,
        patch_size=(2, 4, 4),
        l_dim=128,
        h_dim=384,
        ):
        super().__init__()

        self.l_dim = l_dim

        self.fusion_layer = ResNAF(num_layer=fusion_num_layer, dim=l_dim+h_dim)
        self.low_layer = ResNAF(num_layer=sep_num_layer, dim=l_dim)
        self.high_layer = ResNAF(num_layer=sep_num_layer, dim=h_dim)

        self.unpatch_out = DualFusionUpBlock(
            in_channels_low=l_dim,
            in_channels_high=h_dim,
            out_channels_low=3,
            out_channels_high=21,
            patch_size=patch_size
        )

    def forward(self, x):
        x = self.fusion_layer(x.contiguous())
        x_low = x[:, :self.l_dim] # bCthw
        x_high = x[:, self.l_dim:]

        x_low = self.low_layer(x_low.contiguous())
        x_high = self.high_layer(x_high.contiguous())
        x = self.unpatch_out(x_low, x_high)
        return x