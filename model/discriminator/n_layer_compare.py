import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.discriminator.n_layer import ActNorm, weights_init


class NLayerDiscriminatorCompare(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix."""
    # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
        """
        super(NLayerDiscriminatorCompare, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        input_nc *= 2 # for compare channel concat

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, target, recon):
        # disc goal is to guess which is the original
        # make sure random gen is correct

        orders = torch.rand(size=(target.shape[0], 1, 1, 1)).to(target.device) < 0.5
        x = torch.where(orders, target, recon)
        y = torch.where(orders, recon, target)
        z = torch.cat((x, y), dim=1)  # B(C)HW

        out = self.main(z)
        
        return torch.where(orders, out, -out)
    

class NLayerDiscriminatorCompare3D(nn.Module):
    """Defines a 3D PatchGAN discriminator as in Pix2Pix but for 3D inputs."""
    # https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/opensora/models/causalvideovae/model/losses/discriminator.py
    def __init__(self, input_nc=1, ndf=64, n_layers=3, use_actnorm=False):
        """
        Construct a 3D PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input volumes
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            use_actnorm (bool) -- flag to use actnorm instead of batchnorm
        """
        super(NLayerDiscriminatorCompare3D, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm3d
        else:
            raise NotImplementedError("Not implemented.")
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm3d
        else:
            use_bias = norm_layer != nn.BatchNorm3d

        input_nc *= 2

        kw = 3
        padw = 1
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv3d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=(kw, kw, kw),
                    stride=(2 if n == 1 else 1, 2, 2),
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv3d(
                ndf * nf_mult_prev, ndf * nf_mult, kernel_size=(kw, kw, kw), stride=1, padding=padw, bias=use_bias
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, target, recon):
        orders = torch.rand(size=(target.shape[0], 1, 1, 1, 1)).to(target.device) < 0.5
        x = torch.where(orders, target, recon)
        y = torch.where(orders, recon, target)
        z = torch.cat((x, y), dim=1)  # B(C)THW

        out = self.main(z)
        
        return torch.where(orders, out, -out)