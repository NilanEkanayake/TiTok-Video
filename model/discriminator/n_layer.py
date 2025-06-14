import functools

import torch
import torch.nn as nn

from torch.nn.utils.parametrizations import spectral_norm


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class NLayerDiscriminatorSpectral(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix."""
    # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    def __init__(self, input_nc=3, ndf=64, n_layers=3, output_nc=1):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
        """
        super(NLayerDiscriminatorSpectral, self).__init__()

        use_bias = True

        kw = 4
        padw = 1
        sequence = [
            spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias)),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            spectral_norm(nn.Conv2d(ndf * nf_mult, output_nc, kernel_size=kw, stride=1, padding=padw))
        ]
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)
    

class NLayerDiscriminatorSpectral3D(nn.Module):
    """Defines a 3D PatchGAN discriminator as in Pix2Pix but for 3D inputs."""
    # https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/opensora/models/causalvideovae/model/losses/discriminator.py
    def __init__(self, input_nc=1, ndf=64, n_layers=3, output_nc=1):
        """
        Construct a 3D PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input volumes
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            use_actnorm (bool) -- flag to use actnorm instead of batchnorm
        """
        super(NLayerDiscriminatorSpectral3D, self).__init__()

        use_bias = True

        kw = 3
        padw = 1
        sequence = [
            spectral_norm(nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                spectral_norm(nn.Conv3d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=(kw, kw, kw),
                    stride=(2 if n == 1 else 1, 2, 2),
                    padding=padw,
                    bias=use_bias,
                )),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            spectral_norm(nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=(kw, kw, kw), stride=1, padding=padw, bias=use_bias)),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            spectral_norm(nn.Conv3d(ndf * nf_mult, output_nc, kernel_size=kw, stride=1, padding=padw))
        ]  # output output_nc channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)