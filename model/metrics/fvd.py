"""
Modified from: https://github.com/philippe-eecs/vitok/blob/f415c36b760e96c8b7690576c80291b7a9df2aa4/vitok/evaluators/metrics.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from typing import Tuple
from scipy.linalg import sqrtm

import requests
from tqdm import tqdm

def download(url, local_path, chunk_size=1024):
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)

class FVDCalculator(nn.Module):
    def __init__(self, device='cuda:0'):
        super().__init__()
        detector_url = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'
        local_file_path = 'model/metrics/i3d_torchscript.pt'
        #Download URL
        if not os.path.exists(local_file_path): # no md5 check?
            download(detector_url, local_file_path)

        with open(local_file_path, 'rb') as f:
            self.detector = torch.jit.load(f).eval().to(device, torch.float32)
        self.detector_kwargs = dict(rescale=False, resize=False, return_features=True)

        self.metric_name = 'fvd'
        self.reset()
            
    def reset(self):
        self.fvd_fake_activations = []
        self.fvd_real_activations = []


    def repeat_to_10_frames(self, video_tensor):
        # video_tensor: (b, c, t, h, w)
        b, c, t, h, w = video_tensor.shape
        if t >= 10:
            return video_tensor
        else:
            repeated_tensor = torch.cat([video_tensor, video_tensor[:, :, -1:].repeat(1, 1, 10 - t, 1, 1)], dim=2)
            return repeated_tensor
    
    @torch.no_grad()
    def update(self, recon, target): # BCTHW range -1, 1 in
        generated = recon.float()
        real = target.float()
        
        real = F.interpolate(real, size=(real.shape[1], 224, 224), mode='trilinear', align_corners=False)
        generated = F.interpolate(generated, size=(generated.shape[1], 224, 224), mode='trilinear', align_corners=False)

        # add temporal padding to min 10?
        # replicate last N frames to equal 10?

        real = self.repeat_to_10_frames(real)
        generated = self.repeat_to_10_frames(generated)

        fvd_feats_real = self.detector(real, **self.detector_kwargs)
        fvd_feats_fake = self.detector(generated, **self.detector_kwargs)
        self.fvd_real_activations.append(fvd_feats_real)
        self.fvd_fake_activations.append(fvd_feats_fake)

    def gather(self):
        fvd_real_activations = torch.cat(self.fvd_real_activations, dim=0).cpu().float().numpy()
        fvd_fake_activations = torch.cat(self.fvd_fake_activations, dim=0).cpu().float().numpy()
        # stats = calculate_fid(fvd_fake_activations, fvd_real_activations)
        stats = frechet_distance(fvd_fake_activations, fvd_real_activations)

        return stats
        
    def forward(self):
        pass
    
"""
FROM: https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/opensora/models/causalvideovae/eval/fvd/styleganv/fvd.py
"""

def compute_stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = feats.mean(axis=0) # [d]
    sigma = np.cov(feats, rowvar=False) # [d, d]
    return mu, sigma

def frechet_distance(feats_fake: np.ndarray, feats_real: np.ndarray) -> float:
    mu_gen, sigma_gen = compute_stats(feats_fake)
    mu_real, sigma_real = compute_stats(feats_real)
    m = np.square(mu_gen - mu_real).sum()
    if feats_fake.shape[0]>1:
        s, _ = sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
        fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    else:
        fid = np.real(m)
    return float(fid)