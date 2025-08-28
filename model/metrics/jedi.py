"""
https://github.com/oooolga/JEDi
"""
import sys
sys.path.append("jepa/")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import yaml
import requests
from tqdm import tqdm

from sklearn import metrics
from torchvision.transforms import v2, functional

from jepa.evals.video_classification_frozen.utils import ClipAggregation
from jepa.evals.video_classification_frozen.eval import init_model
from jepa.src.models.attentive_pooler import AttentiveClassifier


VJEPA_CKPT_URLS = {
    'vit_large': {
        'config': 'https://raw.githubusercontent.com/facebookresearch/jepa/refs/heads/main/configs/evals/vitl16_ssv2_16x2x3.yaml',
        'model_ckpt': 'https://dl.fbaipublicfiles.com/jepa/vitl16/vitl16.pth.tar',
        'probe_ckpt': 'https://dl.fbaipublicfiles.com/jepa/vitl16/ssv2-probe.pth.tar',
    },
    'vit_huge': {
        'config': 'https://raw.githubusercontent.com/facebookresearch/jepa/refs/heads/main/configs/evals/vith16_ssv2_16x2x3.yaml',
        'model_ckpt': 'https://dl.fbaipublicfiles.com/jepa/vith16/vith16.pth.tar',
        'probe_ckpt': 'https://dl.fbaipublicfiles.com/jepa/vith16/ssv2-probe.pth.tar',
    },
}

def download(url, local_path, chunk_size=1024):
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


class JEDiMetric(nn.Module):
    def __init__(self, model_name='vit_large', finetuned=True, save_dir='jepa/ckpt', device='cuda:0'):
        super().__init__()
        self.finetuned = finetuned

        model_url = VJEPA_CKPT_URLS[model_name]['model_ckpt']
        probe_url = VJEPA_CKPT_URLS[model_name]['probe_ckpt']
        config_url = VJEPA_CKPT_URLS[model_name]['config']

        os.makedirs(save_dir, exist_ok=True)

        model_path = os.path.join(save_dir, model_name + '.pth.tar')
        probe_path = os.path.join(save_dir, model_name + '-ssv2-probe.pth.tar')
        config_path = os.path.join(save_dir, model_name + '.yaml')

        if not os.path.exists(model_path): # no md5 check?
            download(model_url, model_path)

        if not os.path.exists(probe_path) and finetuned:
            download(probe_url, probe_path)

        if not os.path.exists(config_path):
            download(config_url, config_path)

        with open(config_path, 'r') as y_file:
            config = yaml.load(y_file, Loader=yaml.FullLoader)

        self.model = init_model(
            "cpu",
            model_path,
            model_name,
            patch_size=config['pretrain']['patch_size'],
            crop_size=config['optimization']['resolution'],
            frames_per_clip=config['pretrain']['frames_per_clip'],
            tubelet_size=config['pretrain']['tubelet_size'],
            use_sdpa=config['pretrain']['use_sdpa'],
            use_SiLU=config['pretrain']['use_silu'],
            tight_SiLU=config['pretrain']['tight_silu'],
            uniform_power=config['pretrain']['uniform_power'],
        )

        self.model = ClipAggregation(self.model, tubelet_size=config['pretrain']['tubelet_size'], attend_across_segments=config['optimization']['attend_across_segments'])
        self.model = self.model.eval().to(device, torch.float32)

        if finetuned:
            self.classifier = AttentiveClassifier(embed_dim=self.model.embed_dim, num_heads=self.model.num_heads, depth=1, num_classes=config['data']['num_classes'])
            classifier_sd = {k.replace('module.', ''): v for k, v in torch.load(probe_path)['classifier'].items()}
            self.classifier.load_state_dict(classifier_sd)
            self.classifier = self.classifier.eval().to(device, torch.float32)

        for p in self.parameters():
            p.requires_grad = False

        self.frames_per_clip = config['pretrain']['frames_per_clip']

        self.resize = v2.Resize(size=config['optimization']['resolution'], interpolation=functional.InterpolationMode.BICUBIC, antialias=False)
        self.normalize = v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        self.metric_name = 'jedi'
        self.reset()
            
    def reset(self):
        self.recon_feats = []
        self.target_feats = []


    def pad_frames(self, video_tensor):
        # video_tensor: (b, c, t, h, w)
        b, c, t, h, w = video_tensor.shape
        if t >= self.frames_per_clip:
            return video_tensor
        else:
            repeated_tensor = torch.cat([video_tensor, video_tensor[:, :, -1:].repeat(1, 1, self.frames_per_clip - t, 1, 1)], dim=2)
            return repeated_tensor
    
    def update(self, recon, target): # BCTHW range -1, 1 in
        with torch.no_grad():
            recon_feat = self.get_feats(recon)
            target_feat = self.get_feats(target)

        self.recon_feats.append(recon_feat)
        self.target_feats.append(target_feat)

    def get_feats(self, videos):
        videos = (videos.clamp(-1, 1) + 1) / 2 # [-1, 1] -> [0, 1]

        videos = videos.permute(0, 2, 1, 3, 4)
        videos = self.resize(videos)
        videos = self.normalize(videos)
        videos = videos.permute(0, 2, 1, 3, 4)

        videos = self.pad_frames(videos)

        feats = self.model([[videos]])[0]
        if self.finetuned: # if using ssv2 probe
            return self.classifier.pooler(feats).squeeze(1)
        else:
            return feats.mean(dim=1)

    def gather(self):
        recon_feats = torch.cat(self.recon_feats, dim=0).cpu().float().numpy()
        target_feats = torch.cat(self.target_feats, dim=0).cpu().float().numpy()
        stats = mmd_poly(target_feats, recon_feats, degree=2, coef0=0)*100

        return stats
            
    def forward(self):
        pass

    
"""
FROM: https://github.com/oooolga/JEDi/blob/main/videojedi/mmd_polynomial.py
"""
def mmd_poly(X, Y, degree=2, gamma=None, coef0=0):
    """MMD using polynomial kernel (i.e., k(x,y) = (gamma <X, Y> + coef0)^degree)

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Keyword Arguments:
        degree {int} -- [degree] (default: {2})
        gamma {int} -- [gamma] (default: {1})
        coef0 {int} -- [constant item] (default: {0})

    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.polynomial_kernel(X, X, degree, gamma, coef0)
    YY = metrics.pairwise.polynomial_kernel(Y, Y, degree, gamma, coef0)
    XY = metrics.pairwise.polynomial_kernel(X, Y, degree, gamma, coef0)
    return XX.mean() + YY.mean() - 2 * XY.mean()