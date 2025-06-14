"""
https://github.com/westlake-repl/LeanVAE/blob/master/LeanVAE/utils/patcher_utils.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

_PERSISTENT = False


class Patcher(nn.Module):
    def __init__(self, rescale = True):
        super().__init__()
        self.register_buffer(
            "wavelets", torch.tensor([0.7071067811865476, 0.7071067811865476]), persistent=_PERSISTENT
        )
        self.register_buffer(
            "_arange",
            torch.arange(2),
            persistent=_PERSISTENT,
        )

        self.rescale = rescale
        for param in self.parameters():
            param.requires_grad = False
    
    def _3ddwt(self, x, mode="reflect", rescale=False):
        dtype = x.dtype
        h = self.wavelets

        n = h.shape[0]
        g = x.shape[1]
        hl = h.flip(0).reshape(1, 1, -1).repeat(g, 1, 1)
        hh = (h * ((-1) ** self._arange)).reshape(1, 1, -1).repeat(g, 1, 1)
        hh = hh.to(dtype=dtype)
        hl = hl.to(dtype=dtype)

        # Handles temporal axis.
        x = F.pad(
            x, pad=(max(0, n - 2), n - 1, n - 2, n - 1, n - 2, n - 1), mode=mode
        ).to(dtype)
        xl = F.conv3d(x, hl.unsqueeze(3).unsqueeze(4), groups=g, stride=(2, 1, 1))
        xh = F.conv3d(x, hh.unsqueeze(3).unsqueeze(4), groups=g, stride=(2, 1, 1))

        # Handles spatial axes.
        xll = F.conv3d(xl, hl.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xlh = F.conv3d(xl, hh.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xhl = F.conv3d(xh, hl.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xhh = F.conv3d(xh, hh.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))

        xlll = F.conv3d(xll, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xllh = F.conv3d(xll, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xlhl = F.conv3d(xlh, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xlhh = F.conv3d(xlh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhll = F.conv3d(xhl, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhlh = F.conv3d(xhl, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhhl = F.conv3d(xhh, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhhh = F.conv3d(xhh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))

        out = torch.cat([xlll, xllh, xlhl, xlhh, xhll, xhlh, xhhl, xhhh], dim=1)
        if rescale:
            out = out * (2 * torch.sqrt(torch.tensor(2.0)))
        return out

    def forward(self, x): # excepts -0.5, 0.5 in?
        xv = self._3ddwt(x, rescale=self.rescale)
        return xv


class UnPatcher(nn.Module):

    def __init__(self, rescale = True):
        super().__init__()
        self.register_buffer(
            "wavelets", torch.tensor([0.7071067811865476, 0.7071067811865476]), persistent=_PERSISTENT
        )
        self.register_buffer(
            "_arange",
            torch.arange(2),
            persistent=_PERSISTENT,
        )
        self.rescale = rescale
        for param in self.parameters():
            param.requires_grad = False
    
    def _3didwt(self, x, mode="reflect", rescale=False):
        dtype = x.dtype
        h = self.wavelets
        n = h.shape[0]

        g = x.shape[1] // 8  # split into 8 spatio-temporal filtered tesnors.
        hl = h.flip([0]).reshape(1, 1, -1).repeat([g, 1, 1])
        hh = (h * ((-1) ** self._arange)).reshape(1, 1, -1).repeat(g, 1, 1)
        hl = hl.to(dtype=dtype)
        hh = hh.to(dtype=dtype)

        xlll, xllh, xlhl, xlhh, xhll, xhlh, xhhl, xhhh = torch.chunk(x, 8, dim=1)

        # Height height transposed convolutions.
        xll = F.conv_transpose3d(
            xlll, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2)
        )
        xll += F.conv_transpose3d(
            xllh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2)
        )

        xlh = F.conv_transpose3d(
            xlhl, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2)
        )
        xlh += F.conv_transpose3d(
            xlhh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2)
        )

        xhl = F.conv_transpose3d(
            xhll, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2)
        )
        xhl += F.conv_transpose3d(
            xhlh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2)
        )

        xhh = F.conv_transpose3d(
            xhhl, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2)
        )
        xhh += F.conv_transpose3d(
            xhhh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2)
        )

        # Handles width transposed convolutions.
        xl = F.conv_transpose3d(
            xll, hl.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1)
        )
        xl += F.conv_transpose3d(
            xlh, hh.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1)
        )
        xh = F.conv_transpose3d(
            xhl, hl.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1)
        )
        xh += F.conv_transpose3d(
            xhh, hh.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1)
        )

        # Handles time axis transposed convolutions.
        x = F.conv_transpose3d(
            xl, hl.unsqueeze(3).unsqueeze(4), groups=g, stride=(2, 1, 1)
        )
        x += F.conv_transpose3d(
            xh, hh.unsqueeze(3).unsqueeze(4), groups=g, stride=(2, 1, 1)
        )

        if rescale:
            x = x / (2 * torch.sqrt(torch.tensor(2.0)))
        return x
    
    
    def forward(self, x):
        return self._3didwt(x, rescale=self.rescale)

