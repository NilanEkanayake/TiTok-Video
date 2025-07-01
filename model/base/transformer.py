import torch
import torch.nn as nn
import torch.nn.functional as F

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
        nn.Linear(dim, inner_dim * 2, bias=False),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias=False),
        # nn.LayerNorm(dim), # another LN to fix instability
    )


class Attn(nn.Module):
    def __init__(
        self,
        embed_dim=512,
        num_head=8,
        ):
        super(Attn, self).__init__()

        self.ln = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_head, batch_first=True)

    def forward(self, x):
        x = self.ln(x)
        x = self.attn(query=x, key=x, value=x, need_weights=False)[0]
        return x


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
            # self.attn_layer.append(nn.MultiheadAttention(embed_dim, num_head, batch_first=True))
            self.attn_layer.append(Attn(embed_dim, num_head))
            self.ffd_layer.append(ffd(embed_dim, mlp_ratio)) 
   
    def forward(self, x):
        for i in range(self.num_layer):
            x = x + self.attn_layer[i](x) #(query=x, key=x, value=x, need_weights=False)[0]
            x = x + self.ffd_layer[i](x) 
        return x
