import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model,
            n_head,
            mlp_ratio = 4.0,
            act_layer = nn.GELU,
            norm_layer = nn.LayerNorm,
        ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)

        self.mlp_ratio = mlp_ratio
        # optionally we can disable the FFN
        if mlp_ratio > 0:
            self.ln_2 = norm_layer(d_model)
            mlp_width = int(d_model * mlp_ratio)
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ("gelu", act_layer()),
                ("c_proj", nn.Linear(mlp_width, d_model))
            ]))
            # self.mlp = SwiGLUFFN(d_model, mlp_width)

    def forward(self, x: torch.Tensor):
        residual = x
        x = self.ln_1(x)
        x, _ = self.attn(query=x, key=x, value=x, need_weights=False)
        x = x + residual

        if self.mlp_ratio > 0:
            x = x + self.mlp(self.ln_2(x))

        return x
    
class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim,
        multiple_of=256,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    
class PackedSwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim,
        multiple_of=256,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w13 = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        x1, x3 = torch.chunk(self.w13(x), 2, dim=-1)
        return self.w2(F.silu(x1) * x3)
    