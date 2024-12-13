import torch.nn as nn
from .normalize import Normalize
from .conv import CausalConv3d
import torch
import torch.nn.functional as F

class AttnBlock2D(nn.Module):
    def __init__(self, in_channels, norm_type="groupnorm"):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels, norm_type=norm_type)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class AttnBlock3D(nn.Module):
    """Compatible with old versions, there are issues, use with caution."""

    def __init__(self, in_channels, norm_type="groupnorm"):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels, norm_type=norm_type)
        self.q = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.k = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.v = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.proj_out = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, t, h, w = q.shape
        q = q.reshape(b * t, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b * t, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b * t, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, t, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class AttnBlock3DFix(nn.Module):

    def __init__(self, in_channels, norm_type="groupnorm"):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels, norm_type=norm_type)
        self.q = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.k = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.v = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.proj_out = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, t, h, w = q.shape
        q = q.permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c).contiguous()
        k = k.permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c).contiguous()
        v = v.permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c).contiguous()

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
        )

        attn_output = attn_output.reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3)
        h_ = self.proj_out(attn_output)

        return x + h_