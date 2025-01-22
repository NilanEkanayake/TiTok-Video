import torch
import torch.nn as nn
from collections import OrderedDict

class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model,
            n_head,
            mlp_ratio = 4.0,
            act_layer = nn.GELU,
            norm_layer = nn.LayerNorm,
            exp_res = False,
        ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)

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

        self.exp_res = exp_res

    def attention(
            self,
            x: torch.Tensor
    ):
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(
            self,
            x: torch.Tensor,
    ):
        x = x.permute(1, 0, 2) # NLD -> LND
        residual = x
        x = self.attention(x=self.ln_1(x))

        if not self.exp_res:
            x = x + residual

        if self.mlp_ratio > 0:
            x = x + self.mlp(self.ln_2(x))

        if self.exp_res:
            x = x + residual
        x = x.permute(1, 0, 2) # LND -> NLD

        return x



