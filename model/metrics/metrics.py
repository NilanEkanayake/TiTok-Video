"""
From: https://github.com/philippe-eecs/vitok/blob/main/pytorch_vitok_old/evaluators/metrics.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights
import numpy as np
from scipy import linalg
from sklearn import metrics
from torchmetrics.functional import (
    peak_signal_noise_ratio as PSNR,
    structural_similarity_index_measure as SSIM,)   

class MetricCalculator(nn.Module):
    def __init__(self, metrics=('fid', 'ssim', 'psnr'), log_prefix='eval'):
        super().__init__()
        if 'fid' in metrics or 'is' in metrics or 'mmd' in metrics:
            self.inception = InceptionV3() #.to(torch.float64)
            self.inception.eval()
            # self.transform = nn.Upsample(size=(299, 299), mode="bilinear", align_corners=True)

        self.metrics = metrics
        self.log_prefix = log_prefix
        self.reset()
            
    def reset(self):
        self.fid_fake_activations = []
        self.fid_real_activations = []
        self.inception_score = []
        self.ssim_stats = []
        self.psnr_stats = []

    @torch.no_grad()
    def forward(self, real, generated):
        # list of real/fake images
        # range = [-1, 1]

        for x, y in zip(real, generated):
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

            if 'is' in self.metrics or 'fid' in self.metrics or 'mmd' in self.metrics:
                with torch.autocast(device_type=x.device.type, enabled=False):
                    real_activations, _ = self.inception(x.to(torch.float32)) # 1CHW, [-1, 1] in
                    fake_activations, fake_probs = self.inception(y.to(torch.float32))

                if 'fid' in self.metrics or 'mmd' in self.metrics:
                    self.fid_real_activations.append(real_activations)
                    self.fid_fake_activations.append(fake_activations)
                
                if 'is' in self.metrics:
                    self.inception_score.append(fake_probs)

            if 'ssim' in self.metrics:
                self.ssim_stats.append(SSIM(preds=y, target=x, reduction="elementwise_mean", data_range=(-1, 1)).mean())
            
            if 'psnr' in self.metrics:
                self.psnr_stats.append(PSNR(preds=y, target=x, reduction="elementwise_mean", data_range=(-1, 1)).mean())

        # if 'is' in self.metrics or 'fid' in self.metrics:
        #     real = torch.cat([self.transform(im.unsqueeze(0)) for im in real], dim=0) # resize and batch - crop first?
        #     generated = torch.cat([self.transform(im.unsqueeze(0)) for im in generated], dim=0)

        #     with torch.autocast(device_type=real.device.type, enabled=False): # needed? high mem?
        #         real = real.to(torch.float32)
        #         generated = generated.to(torch.float32)
        #         real_activations, _ = self.inception(real) # BCHW, [-1, 1] in
        #         fake_activations, fake_probs = self.inception(generated)

        #         if 'fid' in self.metrics:
        #             self.fid_real_activations.append(real_activations)
        #             self.fid_fake_activations.append(fake_activations)
                
        #         if 'is' in self.metrics:
        #             self.inception_score.append(fake_probs)

    def gather(self):
        stats = {}     
        if 'fid' in self.metrics or 'mmd' in self.metrics:
            fid_real_activations = torch.cat(self.fid_real_activations, dim=0).cpu().float().numpy() # cat all batches
            fid_fake_activations = torch.cat(self.fid_fake_activations, dim=0).cpu().float().numpy()

            if 'fid' in self.metrics:
                stats['fid'] = torch.tensor(calculate_fid(fid_real_activations, fid_fake_activations)).mean()

            if 'mmd' in self.metrics:
                stats['mmd'] = torch.tensor(mmd_poly(fid_real_activations, fid_fake_activations, degree=2, coef0=0)*100).mean()
        
        if 'is' in self.metrics:
            inception_score = torch.cat(self.inception_score, dim=0).cpu().float().numpy()
            stats['is'] = torch.tensor(compute_inception_score(inception_score)).mean()

        if 'ssim' in self.metrics:
            stats['ssim'] = torch.stack(self.ssim_stats, dim=0).mean()

        if 'psnr' in self.metrics:
            stats['psnr'] = torch.stack(self.psnr_stats, dim=0).mean()

        return {f"{self.log_prefix}/{k}": v for k, v in stats.items()}

def compute_inception_score(softmax_outputs):
    p_yx = softmax_outputs
    p_y = np.mean(p_yx, axis=0)
    kl_div = p_yx * (np.log(p_yx) - np.log(p_y))
    kl_mean = np.mean(np.sum(kl_div, axis=1))
    return float(np.exp(kl_mean))


"""
FROM: https://github.com/oooolga/JEDi/blob/main/videojedi/mmd_polynomial.py

Slightly worse with I3D, much better with V-JEPA and VideoMAE.
Use with transformer-based model then. such as image JEPA?
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
    return float(XX.mean() + YY.mean() - 2 * XY.mean())


def calculate_fid(real_activations, fake_activations):
    mu1, sigma1 = calculate_activation_statistics(real_activations)
    mu2, sigma2 = calculate_activation_statistics(fake_activations)
    fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid_value

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6): # SLOW!
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def calculate_activation_statistics(act):
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

class InceptionV3(nn.Module):
    def __init__(self):
        super(InceptionV3, self).__init__()
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = False
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c
        self.fc = model.fc

    def forward(self, x):
        x = nn.Upsample(size=(299, 299), mode="bilinear", align_corners=True)(x)
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = F.avg_pool2d(x, kernel_size=8)
        acts = x.view(x.size(0), -1)
        acts = F.dropout(acts, training=False)
        logits = self.fc(acts)
        return acts, F.softmax(logits, dim=1)