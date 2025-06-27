"""Stripped version of https://github.com/IntelLabs/cgvqm"""

import torch
import pickle

from torchvision import models
from torchvision.transforms import v2, functional
from torch import nn

def fdiff(a,b,w, normalize_channel=True):
    eps=1e-10
    if normalize_channel:
        norm_a = a.pow(2).sum(dim=1,keepdim=True).sqrt()
        a = a/(norm_a+eps)
        norm_b = b.pow(2).sum(dim=1,keepdim=True).sqrt()
        b = b/(norm_b+eps)
    diff = (w*((a - b)).pow(2).mean([2,3,4],keepdim=True)).sum()

    return diff

class CGVQM(torch.nn.Module):
    def __init__(self, weights_file='cgvqm_weights/cgvqm-2.pickle', num_layers=3): # 'weights/cgvqm-2.pickle', 3 | 'weights/cgvqm-5.pickle', 6
        super().__init__()
        self.model = models.video.r3d_18(weights=models.video.R3D_18_Weights.KINETICS400_V1)

        self.chns = [3,64,64,128,256,512][:num_layers]

        self.loaded_feature_weights = nn.Parameter(torch.ones(1,sum(self.chns),1,1,1))
        self.alpha = nn.Parameter(torch.tensor(1.))

        self.num_layers = num_layers

        with open(weights_file, 'rb') as fp:
            wo,ao = pickle.load(fp) # nosec
            assert(sum(self.chns)==wo.shape[1])
            self.loaded_feature_weights.data = wo
            self.alpha.data = ao

        for param in self.parameters():
            param.requires_grad = False

        self.resize = v2.Resize(size=112, interpolation=functional.InterpolationMode.BILINEAR, antialias=False)
        self.normalize = v2.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])

        self.layers = [self.model.stem, self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4]
        self.feature_weights = torch.split(self.loaded_feature_weights.abs(), self.chns, dim=1)

    def preprocess(self, x):
        x = (x.clamp(-1, 1) + 1) / 2 # [-1, 1] -> [0, 1]

        x = x.permute(0, 2, 1, 3, 4) # BCTHW -> BTCHW
        # x = self.resize(x)
        x = self.normalize(x)
        x = x.permute(0, 2, 1, 3, 4) # BTCHW -> BCTHW

        return x
    
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = self.preprocess(x)
        y = self.preprocess(y)

        diff = fdiff(x, y, self.feature_weights[0].to(x), False)

        for i in range(self.num_layers - 1):
            x, y = self.layers[i](x), self.layers[i](y)
            diff += fdiff(x, y, self.feature_weights[i+1].to(x))

        return diff*100