import torch
import torch.nn as nn
import torch.nn.functional as F

from model.metrics import fvd
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.transforms import v2
from einops import rearrange


class EvalMetrics(nn.Module):
    def __init__(self, config, eval_prefix='eval'):
        super().__init__()
        self.eval_prefix = eval_prefix

        self.metrics = {}
        for m in config.training.eval.log_metrics:
            if m == 'psnr':
                self.metrics[m] = PeakSignalNoiseRatio(data_range=2), 'image' # -1, 1
            elif m == 'ssim':
                self.metrics[m] = StructuralSimilarityIndexMeasure(data_range=2), 'image'
            elif m == 'fvd':
                self.metrics[m] = fvd.FVDCalculator(), 'video'
            elif m == 'jedi':
                from model.metrics import jedi
                self.metrics[m] = jedi.JEDiMetric(model_name=config.training.eval.jedi_jepa_model), 'video'

    
    def update(self, recon, target):
        for m in self.metrics.keys(): # or use autocast? Efficiency doesn't matter as much since eval is lightweight/sporadic.
            if self.metrics[m][0].device != target[0].device:
                self.metrics[m][0].to(target[0].device)

        for x, y in zip(recon, target):
            x = x.clamp(-1, 1)
            for metric, t in self.metrics.values():
                if t == 'image':
                    metric.update(x.transpose(0, 1), y.transpose(0, 1)) # CTHW -> TCHW (T becomes B)
                elif t == 'video':
                    metric.update(x.unsqueeze(0), y.unsqueeze(0)) # CTHW -> BCTHW
                    

    def compute(self):
        out_dict = {}
        for m, (metric, t) in self.metrics.items():
            out_dict[f"{self.eval_prefix}/{m}"] = metric.compute()
        return out_dict
    
    
    def reset(self):
        for metric, t in self.metrics.values():
            metric.reset()