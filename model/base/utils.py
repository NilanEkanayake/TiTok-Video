import torch
import torch.nn as nn
from flash_attn.ops.triton.layer_norm import RMSNorm
from functools import partial
from einops import rearrange


def get_model_dims(model_size='tiny', head_dim=64, mlp_ratio=4.0):
    layers = {
        "tiny": 4,
        "small": 8,
        "base": 12,
        "large": 24,
    }[model_size]
    heads = {
        "tiny": [4, 2],
        "small": [8, 2],
        "base": [12, 4],
        "large": [16, 4],
    }[model_size]

    width = int(head_dim*heads[0])
    return width, layers, heads, mlp_ratio


def patch_rearrange(patch_size):
    N = len(patch_size)
    in_dp = " ".join(f"(d{i} p{i})" for i in range(N))
    out_d = " ".join(f"d{i}" for i in range(N))
    out_p = " ".join(f"p{i}" for i in range(N))

    pattern = f"c {in_dp} -> ({out_d}) ({out_p} c)"
    kwargs = {f"p{i}": size for i, size in enumerate(patch_size)}
    return partial(rearrange, pattern=pattern, **kwargs)


def apply_unpatch(tensor, grid, pattern, kwargs):
    kwargs.update({f"d{i}": g for i, g in enumerate(grid)})
    return rearrange(tensor=tensor, pattern=pattern, **kwargs)


def unpatch_rearrange(patch_size):
    N = len(patch_size)
    in_d = " ".join(f"d{i}" for i in range(N))
    in_p = " ".join(f"p{i}" for i in range(N))
    out_dp = " ".join(f"(d{i} p{i})" for i in range(N))

    pattern = f"({in_d}) ({in_p} c) -> c {out_dp}"
    kwargs = {f"p{i}": size for i, size in enumerate(patch_size)}

    return partial(apply_unpatch, pattern=pattern, kwargs=kwargs)


def init_weights(module):
    if isinstance(module, nn.Linear):
        module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm) or isinstance(module, RMSNorm):
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
        if module.weight is not None:
            nn.init.constant_(module.weight, 1.0)
    elif isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)