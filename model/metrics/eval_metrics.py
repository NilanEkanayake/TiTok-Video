import torch
import torch.nn as nn
import torch.nn.functional as F

from model.metrics import fvd
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics import MetricCollection
from torchvision.transforms import v2

from einops import rearrange

class EvalMetrics(nn.Module):
    def __init__(self, config, eval_prefix='eval'):
        super().__init__()
        self.eval_prefix = eval_prefix

        self.eval_metrics = MetricCollection(
            {
                "psnr": PeakSignalNoiseRatio(data_range=2), # -1, 1
                "ssim": StructuralSimilarityIndexMeasure(data_range=2),
            },
            prefix=f"{eval_prefix}/",
        )

        self.optional_metrics = []

        if config.training.eval.log_fvd:
            self.optional_metrics.append(fvd.FVDCalculator())

        if config.training.eval.log_jedi:
            from model.metrics import jedi
            model_name = config.training.eval.jedi_jepa_model
            self.optional_metrics.append(jedi.JEDiMetric(model_name=model_name))
    
    def update(self, recon, target):
        for x, y in zip(recon, target):
            x = x.clamp(-1, 1)
            self.eval_metrics.update(x.transpose(0, 1), y.transpose(0, 1))

            for metric in self.optional_metrics:
                metric.update(x.unsqueeze(0), y.unsqueeze(0)) # CTHW -> BCTHW

    def compute(self):
        out_dict = self.eval_metrics.compute()
        for metric in self.optional_metrics:
            out_dict[f"{self.eval_prefix}/{metric.metric_name}"] = metric.gather()
        return out_dict
    
    def reset(self):
        self.eval_metrics.reset()
        for metric in self.optional_metrics:
            metric.reset()