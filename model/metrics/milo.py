import os
import torch
from torchvision import transforms
import requests
from tqdm import tqdm
import math

"""https://github.com/ugurcogalan06/MILO"""

def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


class ScalerNetwork(torch.nn.Module):
    def __init__(self, chn_mid=32, use_sigmoid=True):
        super(ScalerNetwork, self).__init__()

        layers = [torch.nn.Conv2d(1, chn_mid, 1, stride=1, padding=0, bias=True),]
        layers += [torch.nn.LeakyReLU(0.2,True),]
        layers += [torch.nn.Conv2d(chn_mid, chn_mid, 1, stride=1, padding=0, bias=True),]
        layers += [torch.nn.LeakyReLU(0.2,True),]
        layers += [torch.nn.Conv2d(chn_mid, 1, 1, stride=1, padding=0, bias=True),]
        if(use_sigmoid):
            layers += [torch.nn.Sigmoid(),]
        self.model = torch.nn.Sequential(*layers)

    def forward(self, val):
        return self.model.forward(val)


class MaskFinder(torch.nn.Module):
    def __init__(self, input_channels, num_features=64):
        super(MaskFinder, self).__init__()

        self.netBasic = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)
        )

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, inputChannels):
        out_net = self.netBasic(inputChannels)

        out = self.sigmoid(out_net)

        return out


class MILO(torch.nn.Module):
    def __init__(self):

        super(MILO, self).__init__()
        weights_url = "https://github.com/ugurcogalan06/MILO/raw/refs/heads/main/weights/MILO.pth"
        local_file_path = "model/metrics/MILO.pth"

        if not os.path.exists(local_file_path): # no md5 check?
            download(weights_url, local_file_path)

        # Init All Components
        # self.cuda()
        self.mask_finder_1 = MaskFinder(7) #.cuda()

        self.mask_finder_1.requires_grad = False
        self.number_of_scales = 3

        self.scaler_network = ScalerNetwork()

        self.load_state_dict(torch.load(local_file_path, map_location='cpu'), strict=True)


    def mask_generator(self, y, x):
        B, C, H, W = y.shape[0:4]

        refScale = [x]
        distScale = [y]

        for intLevel in range(self.number_of_scales):
            # if tenOne[0].shape[2] > 32 or tenOne[0].shape[3] > 32:
            refScale.insert(0, torch.nn.functional.avg_pool2d(input=refScale[0], kernel_size=2, stride=2,
                                                              count_include_pad=False))
            distScale.insert(0, torch.nn.functional.avg_pool2d(input=distScale[0], kernel_size=2, stride=2,
                                                               count_include_pad=False))
            # end
        # end

        mask = refScale[0].new_zeros([refScale[0].shape[0], 1, int(math.floor(refScale[0].shape[2] / 2.0)),
                                      int(math.floor(refScale[0].shape[3] / 2.0))])

        for intLevel in range(len(refScale)):
            maskUpsampled = torch.nn.functional.interpolate(input=mask, scale_factor=2, mode='bilinear',
                                                            align_corners=True)

            if maskUpsampled.shape[2] != refScale[intLevel].shape[2]: maskUpsampled = torch.nn.functional.pad(
                input=maskUpsampled, pad=[0, 0, 0, 1], mode='replicate')
            if maskUpsampled.shape[3] != refScale[intLevel].shape[3]: maskUpsampled = torch.nn.functional.pad(
                input=maskUpsampled, pad=[0, 1, 0, 0], mode='replicate')

            mask = self.mask_finder_1(
                torch.cat([refScale[intLevel], distScale[intLevel], maskUpsampled], 1)) + maskUpsampled

        return mask


    def forward(self, y, x):
        x = (x.clamp(-1, 1) + 1) / 2 # [-1, 1] -> [0, 1]
        y = (y.clamp(-1, 1) + 1) / 2
        mask = self.mask_generator(x, y)
        score = ((mask * torch.abs(x - y))).mean() 
        
        return score 
