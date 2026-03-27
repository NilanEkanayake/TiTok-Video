import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_attn.ops.triton.layer_norm import RMSNorm
from xformers.ops import SwiGLU
from flash_attn import flash_attn_varlen_func
from model.base.rope import apply_rotary_emb
from einops import rearrange
import math

"""
Modified from: https://github.com/westlake-repl/LeanVAE/blob/master/LeanVAE/modules/backbones.py
"""

def ffd_swi(dim, mult=4, mult_of=32, dropout=0.):
    inner_dim = int(mult * (2 / 3) * dim)
    inner_dim = mult_of * ((inner_dim + mult_of - 1) // mult_of)
    return nn.Sequential(
        RMSNorm(dim),
        SwiGLU(dim, inner_dim, bias=True),
        nn.Dropout(dropout),
    )


# class GEGLU(nn.Module):
#     def __init__(self, dim, mult=4, mult_of=32, dropout=0.):
#         super().__init__()
#         inner_dim = int(mult * (2 / 3) * dim)
#         inner_dim = mult_of * ((inner_dim + mult_of - 1) // mult_of)

#         self.norm = RMSNorm(dim)
#         self.w12 = nn.Linear(dim, inner_dim * 2, bias=False)
#         self.drop1 = nn.Dropout(dropout)
#         self.w3 = nn.Linear(inner_dim, dim, bias=False)

#     def forward(self, x):
#         x = self.norm(x.contiguous())
#         x = self.w12(x)

#         x, gate = x.chunk(2, dim=-1)
#         x = F.gelu(gate) * x

#         x = self.drop1(x)
#         x = self.w3(x)
#         return x


class Attn(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.dim = dim
        self.q_heads, self.kv_heads = heads
        self.head_dim = dim//self.q_heads
        self.gqa_dim = self.head_dim * self.kv_heads

        self.pre_ln = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, (self.gqa_dim * 2) + dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, freqs, cu_seqlens, max_seqlen):
        x = self.pre_ln(x.contiguous())
        q, k, v = self.to_qkv(x).split([self.dim, self.gqa_dim, self.gqa_dim], dim=-1)

        q = q.unflatten(-1, (self.q_heads, self.head_dim)) # [L, H_Q, D_H]
        k = k.unflatten(-1, (self.kv_heads, self.head_dim)) # [L, H_KV, D_H]
        v = v.unflatten(-1, (self.kv_heads, self.head_dim))

        q = apply_rotary_emb(q, freqs)
        k = apply_rotary_emb(k, freqs)

        x = flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens, max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen)

        x = x.flatten(-2).contiguous() # flatten to [L, D]
        return self.out_proj(x)
    


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            embed_dim=512,
            heads=[8, 2],
            mlp_ratio=4,
            num_layer=2,
        ): 
        super(ResidualAttentionBlock, self).__init__()
        self.num_layer = num_layer
        self.attn_layer = nn.ModuleList([Attn(embed_dim, heads) for _ in range(num_layer)])
        self.ffd_layer = nn.ModuleList([ffd_swi(embed_dim, mlp_ratio) for _ in range(num_layer)])

    def forward(self, x, freqs, cu_seqlens, max_seqlen):
        for i in range(self.num_layer):
            x = x + self.attn_layer[i](x, freqs, cu_seqlens, max_seqlen)
            x = x + self.ffd_layer[i](x.contiguous())
        return x