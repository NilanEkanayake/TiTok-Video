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

# https://arxiv.org/pdf/2602.08626
class SplitNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()

        self.eps = eps
        self.weight_t = nn.Parameter(torch.ones(1, dim))
        self.weight_p = nn.Parameter(torch.ones(1, dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    @torch.compile()
    def forward(self, x, mask):
        x = self._norm(x.float()).type_as(x) # disable autocast? Run all in BF16?
        weight = torch.where(mask.unsqueeze(-1), self.weight_t.type_as(x), self.weight_p.type_as(x))
        x = x * weight
        return x


class GEGLU(nn.Module):
    def __init__(self, dim, mult=4, mult_of=32, dropout=0.):
        super().__init__()
        inner_dim = int(mult * (2 / 3) * dim)
        inner_dim = mult_of * ((inner_dim + mult_of - 1) // mult_of)

        self.norm = RMSNorm(dim)
        self.w12 = nn.Linear(dim, inner_dim * 2, bias=False)
        self.drop1 = nn.Dropout(dropout)
        self.w3 = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)
        x = self.w12(x)

        x, gate = x.chunk(2, dim=-1)
        x = F.gelu(gate) * x

        x = self.drop1(x)
        x = self.w3(x)
        return x


def ffd_swi(dim, mult=4, mult_of=32, dropout=0.):
    inner_dim = int(mult * (2 / 3) * dim)
    inner_dim = mult_of * ((inner_dim + mult_of - 1) // mult_of)
    return nn.Sequential(
        RMSNorm(dim),
        SwiGLU(dim, inner_dim, bias=True),
        nn.Dropout(dropout),
    )


class Attn(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.dim = dim
        self.q_heads, self.kv_heads = heads
        self.head_dim = dim//self.q_heads
        self.gqa_dim = self.head_dim * self.kv_heads

        self.pre_ln = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, (self.gqa_dim * 2) + (dim * 2), bias=False) # add gate (v+k) + (q+gate) dims
        # self.to_qkv = nn.Linear(dim, (self.gqa_dim * 2) + dim, bias=False) # add gate (v+k) + (q+gate) dims

        # self.q_norm = RMSNorm(self.head_dim)
        # self.k_norm = RMSNorm(self.head_dim)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, freqs, cu_seqlens, max_seqlen):
        x = self.pre_ln(x)
        q, gate, k, v = self.to_qkv(x).split([self.dim, self.dim, self.gqa_dim, self.gqa_dim], dim=-1)
        # q, k, v = self.to_qkv(x).split([self.dim, self.gqa_dim, self.gqa_dim], dim=-1)

        q = q.unflatten(-1, (self.q_heads, self.head_dim)) # [L, H_Q, D_H]
        k = k.unflatten(-1, (self.kv_heads, self.head_dim)) # [L, H_KV, D_H]
        v = v.unflatten(-1, (self.kv_heads, self.head_dim))

        # q = self.q_norm(q.contiguous()).to(v) # why is the cast needed?
        # k = self.k_norm(k.contiguous()).to(v)

        q = apply_rotary_emb(q, freqs)
        k = apply_rotary_emb(k, freqs)

        x = flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens, max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen)

        x = x.flatten(-2).contiguous() # flatten to [L, D]
        x = x * torch.sigmoid(gate) # https://arxiv.org/pdf/2505.06708 | gating similar to qwen3-next
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
        self.alpha = num_layer * 2 # x2 since each layer has FFN and Attn sub-layers.

        self.attn_layer = nn.ModuleList([Attn(embed_dim, heads) for _ in range(num_layer)])
        self.ffd_layer = nn.ModuleList([GEGLU(embed_dim, mult=mlp_ratio) for _ in range(num_layer)])
    
        self.attn_post_ln = nn.ModuleList([RMSNorm(embed_dim) for _ in range(num_layer - 1)])
        self.ffd_post_ln = nn.ModuleList([RMSNorm(embed_dim) for _ in range(num_layer - 1)])

   
    def forward(self, x, freqs, cu_seqlens, max_seqlen):
        for i in range(self.num_layer):
            if i==0: # normal pre-ln
                x = x + self.attn_layer[i](x.contiguous(), freqs, cu_seqlens, max_seqlen)
                x = x + self.ffd_layer[i](x.contiguous())
            else:
                # https://arxiv.org/pdf/2601.19895 - aka KEEL
                """
                From paper:
                Setting α = L is critical for maintaining training stability in very large-scale or deep models. For smaller architectures where
                vanishing or exploding gradients are less pronounced, α can be treated as a tunable hyperparameter (α > 1) to potentially
                accelerate convergence

                Thus, [self.num_layer * 2] can be replaced with a manually-set value > 1.
                """
                x = (self.alpha * x) + self.attn_layer[i](x.contiguous(), freqs, cu_seqlens, max_seqlen)
                x = self.attn_post_ln[i-1](x.contiguous())

                x = (self.alpha * x) + self.ffd_layer[i](x.contiguous())
                x = self.ffd_post_ln[i-1](x.contiguous())
        return x