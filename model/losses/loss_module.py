import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np

from model.base.blocks import TiTokEncoder, init_weights
from model.metrics.lpips import LPIPS
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode
import random


def l1(x, y):
    return torch.abs(x - y)

# https://github.com/bytedance/1d-tokenizer/blob/main/modeling/modules/losses.py
def compute_lecam_loss(
    logits_real_mean: torch.Tensor,
    logits_fake_mean: torch.Tensor,
    ema_logits_real_mean: torch.Tensor,
    ema_logits_fake_mean: torch.Tensor
) -> torch.Tensor:
    """Computes the LeCam loss for the given average real and fake logits.

    Args:
        logits_real_mean -> torch.Tensor: The average real logits.
        logits_fake_mean -> torch.Tensor: The average fake logits.
        ema_logits_real_mean -> torch.Tensor: The EMA of the average real logits.
        ema_logits_fake_mean -> torch.Tensor: The EMA of the average fake logits.

    Returns:
        lecam_loss -> torch.Tensor: The LeCam loss.
    """
    lecam_loss = torch.mean(torch.pow(F.relu(logits_real_mean - ema_logits_fake_mean), 2))
    lecam_loss += torch.mean(torch.pow(F.relu(ema_logits_real_mean - logits_fake_mean), 2))
    return lecam_loss

    
class ReconstructionLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.perceptual_weight = config.losses.recon.perceptual_weight
        if self.perceptual_weight > 0.0:
            self.perceptual_model = LPIPS().eval()
            for param in self.perceptual_model.parameters():
                param.requires_grad = False

        self.use_disc = config.losses.disc.use_disc
        self.disc_weight = config.losses.disc.disc_weight
        self.disc_start = config.losses.disc.disc_start

        if self.use_disc:
            self.disc_model = TiTokEncoder( # same arch as tokenizer encoder
                model_size=config.model.disc.model_size,
                patch_size=config.model.disc.patch_size,
                in_channels=3,
                out_channels=1, # more stable to use more channels?
                max_grid=config.training.sampling.max_grid,
                max_tokens=config.training.sampling.num_token_range[1],
            ).apply(init_weights)

            if config.training.main.torch_compile:
                self.disc_model = torch.compile(self.disc_model)

            self.lecam_weight = config.losses.disc.lecam_weight
            self.lecam_decay = config.losses.disc.get("lecam_decay", 0.999)
            if self.lecam_weight > 0.0:
                self.register_buffer("ema_real_logits_mean", torch.zeros((1)))
                self.register_buffer("ema_fake_logits_mean", torch.zeros((1)))

        self.total_steps = config.training.main.max_steps

    
    def perceptual_preprocess(self, recon, target, resize_prob=0.25):
        target_out = []
        recon_out = []
        sample_size = self.config.losses.recon.perceptual_sampling_size

        for trg_frame, rec_frame in zip(target, recon):
            # CHW in
            rec_frame = rec_frame.clamp(-1, 1)

            # random resize
            H, W = trg_frame.shape[1:]
            if (H < sample_size or W < sample_size) or random.random() < resize_prob:
                trg_frame = v2.functional.resize(trg_frame, size=sample_size, interpolation=InterpolationMode.BICUBIC, antialias=False)
                rec_frame = v2.functional.resize(rec_frame, size=sample_size, interpolation=InterpolationMode.BICUBIC, antialias=False)

            H, W = trg_frame.shape[1:]
            height_offset = random.randrange(0, (H-sample_size)+1) # no +1?
            width_offset = random.randrange(0, (W-sample_size)+1)

            trg_frame = trg_frame[:, height_offset:height_offset+sample_size, width_offset:width_offset+sample_size]
            rec_frame = rec_frame[:, height_offset:height_offset+sample_size, width_offset:width_offset+sample_size]

            target_out.append(trg_frame)
            recon_out.append(rec_frame)

        target = torch.stack(target_out, dim=0).contiguous() # now FCHW
        recon = torch.stack(recon_out, dim=0).contiguous()
        return recon, target


    def forward(self, target, recon, global_step, results_dict=None, disc_forward=False):
        if disc_forward:
            return self._forward_discriminator(target, recon, global_step)
        else:
            return self._forward_generator(target, recon, global_step, results_dict)
        

    def _forward_generator(self, target, recon, global_step, results_dict=None):
        # target and recon are now lists of CTHW tensors.
        loss_dict = {}

        target = [i.contiguous() for i in target]
        recon = [i.contiguous() for i in recon]

        B = len(target)

        recon_loss = torch.stack([l1(x, y).mean() for x, y in zip(target, recon)]).mean() # not [B]
        loss_dict['recon_loss'] = recon_loss

        perceptual_loss = 0.0
        if self.perceptual_weight > 0.0:
            num_subsample = self.config.losses.recon.perceptual_samples_per_step

            target_frames = []
            recon_frames = []
            for trg_vid, rec_vid in zip(target, recon):
                target_frames += trg_vid.unbind(1) # unbind T dim
                recon_frames += rec_vid.unbind(1)

            if num_subsample != -1 and num_subsample < len(target_frames):                
                # shuffle identically
                tmp = list(zip(target_frames, recon_frames))
                random.shuffle(tmp)
                target_frames, recon_frames = zip(*tmp)

                target_frames = target_frames[:num_subsample]
                recon_frames = recon_frames[:num_subsample]

            recon_frames, target_frames = self.perceptual_preprocess(recon_frames, target_frames)
            perceptual_loss = self.perceptual_model(recon_frames, target_frames).mean()

            loss_dict['perceptual_loss'] = perceptual_loss

        # adversarial loss - not adapted yet, will need to use ViT for nested support?
        d_weight = self.disc_weight
        d_weight_warm = min(1.0, ((global_step - self.disc_start) / self.config.losses.disc.disc_weight_warmup_steps))
        g_loss = 0.0
        if self.use_disc and global_step > self.disc_start:
            target = [i.detach().contiguous() for i in target]

            ############################
            for param in self.disc_model.parameters():
                param.requires_grad = False
            ############################

            logits_real = self.disc_model(target, token_counts=[1]*B).view(B, -1).mean(1)
            logits_fake = self.disc_model(recon, token_counts=[1]*B).view(B, -1).mean(1)

            logits_relative = logits_fake - logits_real
            g_loss = F.softplus(-logits_relative)

            ######################
            loss_dict['gan_loss'] = g_loss
            loss_dict['logits_relative'] = logits_relative

        total_loss = (
            recon_loss
            + (self.perceptual_weight * perceptual_loss)
            + (d_weight * d_weight_warm * g_loss)
        ).mean()

        loss_dict['total_loss'] = total_loss
            
        return total_loss, {'train/'+k:v.clone().mean().detach() for k,v in loss_dict.items()}
    
    
    def _forward_discriminator(self, target, recon, global_step):
        loss_dict = {}
        
        target = [i.detach().requires_grad_(True).contiguous() for i in target]
        recon = [i.detach().requires_grad_(True).contiguous() for i in recon]

        B = len(target)

        ############################
        for param in self.disc_model.parameters():
            param.requires_grad = True
        ############################

        logits_real = self.disc_model(target, token_counts=[1]*B).view(B, -1).mean(1) # model out = [B*L, C] or [B*1, 1]
        logits_fake = self.disc_model(recon, token_counts=[1]*B).view(B, -1).mean(1)

        # disc model outputs normal dense tensor
 
        logits_relative = logits_real - logits_fake

        lecam_loss = 0.0
        if self.lecam_weight > 0.0:
            lecam_loss = compute_lecam_loss(
                logits_real.mean(),
                logits_fake.mean(),
                self.ema_real_logits_mean,
                self.ema_fake_logits_mean
            )
            self.ema_real_logits_mean = self.ema_real_logits_mean * self.lecam_decay + logits_real.clone().mean().detach() * (1 - self.lecam_decay)
            self.ema_fake_logits_mean = self.ema_fake_logits_mean * self.lecam_decay + logits_fake.clone().mean().detach() * (1 - self.lecam_decay)
            loss_dict['lecam_loss'] = lecam_loss

        disc_loss = F.softplus(-logits_relative).mean() + self.lecam_weight * lecam_loss
        
        loss_dict.update({
            "disc_loss": disc_loss,
            "logits_real": logits_real,
            "logits_fake": logits_fake,
        })
            
        return disc_loss, {'train/'+k:v.clone().mean().detach() for k,v in loss_dict.items()}