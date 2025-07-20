import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base.sigma_reparam import SNLinear
from model.base.rope import apply_rotary_emb
from einops import rearrange

# from timm.layers import Attention

"""
Modified from: https://github.com/westlake-repl/LeanVAE/blob/master/LeanVAE/modules/backbones.py
"""

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x
    
def ffd(dim, mult=4, mult_of=32, dropout=0.):
    inner_dim = int(mult * (2 / 3) * dim)
    inner_dim = mult_of * ((inner_dim + mult_of - 1) // mult_of)
    return nn.Sequential(
        nn.LayerNorm(dim),
        SNLinear(dim, inner_dim * 2, bias=False),
        GEGLU(),
        nn.Dropout(dropout),
        SNLinear(inner_dim, dim, bias=False),
        nn.LayerNorm(dim), # another LN to fix instability
    )

class Attn(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.dim = dim
        self.heads = heads

        # enabling both linears = 10.1m params.
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.q_norm = nn.LayerNorm(dim)
        self.k_norm = nn.LayerNorm(dim)

    def forward(self, x, freqs):
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # apply rope - prior to head split, see LTX-video
        q = apply_rotary_emb(q, freqs)
        k = apply_rotary_emb(k, freqs)

        def split_heads(t):
            return t.unflatten(-1, (self.heads, self.dim//self.heads)).transpose(1, 2)

        q, k, v = [split_heads(i) for i in (q, k, v)]

        out = F.scaled_dot_product_attention(q, k, v, enable_gqa=False, is_causal=False)

        out = out.transpose(1, 2).flatten(-2)

        return self.out_proj(out)


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            embed_dim=512,
            num_head=8,
            mlp_ratio=4,
            num_layer=2,
        ): 
        super(ResidualAttentionBlock, self).__init__()
        self.num_layer = num_layer
        self.attn_layer = nn.Sequential()
        self.ffd_layer = nn.Sequential()
        for _ in range(num_layer):
            self.attn_layer.append(Attn(embed_dim, num_head))
            self.ffd_layer.append(ffd(embed_dim, mlp_ratio)) 
   
    def forward(self, x, freqs):
        for i in range(self.num_layer):
            x = x + self.attn_layer[i](x, freqs)
            x = x + self.ffd_layer[i](x) 
        return x