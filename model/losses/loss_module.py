import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np

from model.base.blocks import TiTokEncoder, init_weights
from model.metrics.lpips import LPIPS
# from model.metrics.milo import MILO
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode
import random


def l1(x, y):
    return torch.abs(x - y)

def lecam_reg(real_pred, fake_pred, ema_real_pred, ema_fake_pred):
    """Lecam loss for data-efficient and stable GAN training.
    
    Described in https://arxiv.org/abs/2104.03310
    
    Args:
      real_pred: Prediction (scalar) for the real samples.
      fake_pred: Prediction for the fake samples.
      ema_real_pred: EMA prediction (scalar) for the real samples.
      ema_fake_pred: EMA prediction for the fake samples.
    
    Returns:
      Lecam regularization loss (scalar).
    """
    assert real_pred.ndim == 0 and ema_fake_pred.ndim == 0
    lecam_loss = torch.mean(torch.pow(torch.relu(real_pred - ema_fake_pred), 2))
    lecam_loss = lecam_loss + torch.mean(torch.pow(torch.relu(ema_real_pred - fake_pred), 2))
    return lecam_loss

    
class ReconstructionLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.perceptual_weight = config.tokenizer.losses.perceptual_weight
        if self.perceptual_weight > 0.0:
            self.perceptual_model = LPIPS().eval()
            # self.perceptual_model = MILO().eval()
            for param in self.perceptual_model.parameters():
                param.requires_grad = False

        loss_d = config.discriminator.losses
        model_d = config.discriminator.model
        self.use_disc = config.discriminator.use_disc
        self.disc_weight = loss_d.disc_weight
        self.disc_start = loss_d.disc_start
        self.disc_warm = max(loss_d.disc_warmup_steps, 1)
        self.disc_weight_warm = loss_d.disc_weight_warmup_steps
        self.token_range = config.training.sampling.num_token_range

        if self.use_disc:
            self.disc_model = TiTokEncoder( # same arch as tokenizer encoder
                model_size=model_d.model_size,
                patch_size=model_d.patch_size,
                in_channels=3,
                out_channels=16, # more stable to use more channels?
                max_grid=config.training.sampling.max_grid,
                max_tokens=self.token_range[1],
            ).apply(init_weights)

            if config.training.main.torch_compile:
                self.disc_model = torch.compile(self.disc_model)

        self.lecam_weight = loss_d.lecam_weight
        if self.lecam_weight > 0.0:
            self.register_buffer('lecam_ema_real', torch.tensor(0.0))
            self.register_buffer('lecam_ema_fake', torch.tensor(0.0))

        self.gradient_penalty_weight = loss_d.gradient_penalty_weight
        self.total_steps = config.training.main.max_steps


    @torch.no_grad()
    def update_lecam_ema(self, real, fake, decay=0.999):
        with torch.autocast(device_type=real.device.type, enabled=False):
            real, fake = real.float().mean(), fake.float().mean()
            self.lecam_ema_real.mul_(decay).add_(real, alpha=1 - decay)
            self.lecam_ema_fake.mul_(decay).add_(fake, alpha=1 - decay)
    

    def perceptual_preprocess(self, recon, target, resize_prob=0.25):
        target_out = []
        recon_out = []
        sample_size = self.config.tokenizer.losses.perceptual_sampling_size

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


    def forward(self, target, recon, global_step, disc_forward=False):
        if disc_forward:
            return self._forward_discriminator(target, recon)
        else:
            return self._forward_generator(target, recon, global_step)
        

    def _forward_generator(self, target, recon, global_step):
        # target and recon are now lists of CTHW tensors.
        loss_dict = {}

        target = [i.contiguous() for i in target]
        recon = [i.contiguous() for i in recon]

        B = len(target)

        recon_loss = torch.stack([l1(x, y).mean() for x, y in zip(target, recon)]).mean() # not [B]
        loss_dict['recon_loss'] = recon_loss

        perceptual_loss = 0.0
        if self.perceptual_weight > 0.0:
            num_subsample = self.config.tokenizer.losses.perceptual_samples_per_step

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


        d_weight = self.disc_weight
        d_weight_warm = min(1.0, ((global_step - (self.disc_start+self.disc_warm)) / self.disc_weight_warm))
        g_loss = 0.0
        if self.use_disc and global_step > self.disc_start+self.disc_warm:
            target = [i.detach().contiguous() for i in target]

            ############################
            for param in self.disc_model.parameters():
                param.requires_grad = False
            ############################

            logits_real = self.disc_model(target, token_counts=[1]*B).view(B, -1).mean(-1)
            logits_fake = self.disc_model(recon, token_counts=[1]*B).view(B, -1).mean(-1)
            logits_relative = logits_fake - logits_real
            g_loss = F.softplus(-logits_relative).mean()

            ######################
            loss_dict['gan_loss'] = g_loss
            loss_dict['d_weight'] = torch.tensor(d_weight * d_weight_warm)
            # loss_dict['logits_relative'] = logits_relative

        total_loss = (
            recon_loss
            + (self.perceptual_weight * perceptual_loss)
            + (d_weight * d_weight_warm * g_loss)
        ).mean()

        loss_dict['total_loss'] = total_loss
            
        return total_loss, {'train/'+k:v.clone().mean().detach() for k,v in loss_dict.items()}
    
    
    def _forward_discriminator(self, target, recon):
        loss_dict = {}
        
        target = [i.detach().requires_grad_(True).contiguous() for i in target]
        recon = [i.detach().requires_grad_(True).contiguous() for i in recon]

        B = len(target)

        ############################
        for param in self.disc_model.parameters():
            param.requires_grad = True
        ############################

        logits_real = self.disc_model(target, token_counts=[1]*B).view(B, -1).mean(-1)
        logits_fake = self.disc_model(recon, token_counts=[1]*B).view(B, -1).mean(-1)
        logits_relative = logits_real - logits_fake
        d_loss = F.softplus(-logits_relative).mean()

        # https://www.arxiv.org/pdf/2509.24935
        gradient_penalty = 0.0
        sigma = self.config.discriminator.losses.gradient_penalty_noise
        if self.gradient_penalty_weight > 0.0:
            noise = [torch.randn_like(x) * sigma for x in target] # diff noise per sample? averages out?
            logits_real_noised = self.disc_model([x + y for x, y in zip(target, noise)], token_counts=[1]*B).view(B, -1).mean(-1)
            logits_fake_noised = self.disc_model([x + y for x, y in zip(recon, noise)], token_counts=[1]*B).view(B, -1).mean(-1)
            r1_penalty = (logits_real - logits_real_noised)**2
            r2_penalty = (logits_fake - logits_fake_noised)**2

            loss_dict['r1_penalty'] = r1_penalty
            loss_dict['r2_penalty'] = r2_penalty
            gradient_penalty = r1_penalty + r2_penalty

        lecam_loss = 0.0
        if self.lecam_weight > 0.0:
            lecam_loss = self.lecam_weight * lecam_reg(
                real_pred=logits_real.mean(),
                fake_pred=logits_fake.mean(),
                ema_real_pred=self.lecam_ema_real,
                ema_fake_pred=self.lecam_ema_fake,
            )
            self.update_lecam_ema(logits_real, logits_fake)
            loss_dict['lecam_loss'] = lecam_loss

        total_loss = (
            d_loss
            + (self.lecam_weight * lecam_loss)
            + (self.gradient_penalty_weight / sigma**2 * gradient_penalty)
        ).mean()
        
        loss_dict.update({
            "disc_loss": total_loss,
            "logits_real": logits_real,
            "logits_fake": logits_fake,
        })
            
        return total_loss, {'train/'+k:v.clone().mean().detach() for k,v in loss_dict.items()}