import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base.sigma_reparam import SNLinear
from flash_attn import flash_attn_varlen_func

from model.base.rope import apply_rotary_emb
from einops import rearrange


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
        self.q_heads, self.kv_heads = heads
        self.head_dim = dim//self.q_heads
        self.gqa_dim = self.head_dim * self.kv_heads

        self.to_qkv = nn.Linear(dim, self.gqa_dim * 2 + dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)

    def forward(self, x, freqs, cu_seqlens, max_seqlen):
        q, k, v = self.to_qkv(x).split([self.dim, self.gqa_dim, self.gqa_dim], dim=-1)

        q = q.unflatten(-1, (self.q_heads, self.head_dim)) # [L, H_Q, D_H]
        k = k.unflatten(-1, (self.kv_heads, self.head_dim)) # [L, H_KV, D_H]
        v = v.unflatten(-1, (self.kv_heads, self.head_dim))

        q = self.q_norm(q).to(v) # why is the cast needed?
        k = self.k_norm(k).to(v)

        q = apply_rotary_emb(q, freqs)
        k = apply_rotary_emb(k, freqs)

        x = flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens, max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen)

        return self.out_proj(x.flatten(-2)) # flatten to [L, D]


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
        self.attn_layer = nn.Sequential()
        self.ffd_layer = nn.Sequential()
        for _ in range(num_layer):
            self.attn_layer.append(Attn(embed_dim, heads))
            self.ffd_layer.append(ffd(embed_dim, mlp_ratio)) 
   
    def forward(self, x, freqs, cu_seqlens, max_seqlen):
        for i in range(self.num_layer):
            x = x + self.attn_layer[i](x.contiguous(), freqs, cu_seqlens, max_seqlen)
            x = x + self.ffd_layer[i](x.contiguous()) 
        return x