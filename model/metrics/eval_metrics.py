import torch
import torch.nn as nn
import torch.nn.functional as F

from model.metrics import jedi, fvd
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics import MetricCollection

from einops import rearrange

class EvalMetrics(nn.Module):
    def __init__(self, config, eval_prefix='eval'):
        super().__init__()
        self.eval_prefix = eval_prefix

        self.eval_metrics = MetricCollection(
            {
                "psnr": PeakSignalNoiseRatio(),
                "ssim": StructuralSimilarityIndexMeasure(),
            },
            prefix=f"{eval_prefix}/",
        )

        self.optional_metrics = []
        if config.training.eval.log_fvd:
            self.optional_metrics.append(fvd.FVDCalculator())

        if config.training.eval.log_jedi:
            model_name = config.training.eval.jedi_jepa_model
            self.optional_metrics.append(jedi.JEDiMetric(model_name=model_name))

    def set_device(self, device):
        for metric in self.optional_metrics:
            metric.set_device(device)

    def update(self, recon, target):
        self.eval_metrics.update(rearrange(recon, "b c t h w -> (b t) c h w"), rearrange(target, "b c t h w -> (b t) c h w"))
        for metric in self.optional_metrics:
            metric.update(recon, target)

    def compute(self):
        out_dict = self.eval_metrics.compute()
        for metric in self.optional_metrics:
            out_dict[f"{self.eval_prefix}/{metric.metric_name}"] = metric.gather()
        return out_dict
    
    def reset(self):
        self.eval_metrics.reset()
        for metric in self.optional_metrics:
            metric.reset()