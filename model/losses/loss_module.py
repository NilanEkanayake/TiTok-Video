import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np

from model.base.blocks import TiTokEncoder
from model.base.utils import init_weights

from model.metrics.lpips_gram import LPIPS
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode
import random


def l1(x, y):
    return torch.abs(x - y)

    
class ReconstructionLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        loss_c = config.tokenizer.losses
        loss_d = config.discriminator.losses

        self.perceptual_weight = loss_c.perceptual_weight
        self.gram_weight = loss_c.gram_weight
        if self.perceptual_weight > 0.0 or self.gram_weight > 0.0:
            self.perceptual_model = LPIPS().eval()
            for param in self.perceptual_model.parameters():
                param.requires_grad = False

            if config.training.main.torch_compile:
                self.perceptual_model = torch.compile(self.perceptual_model)
        
        model_d = config.discriminator.model
        self.disc_weight = loss_c.disc_weight

        if self.disc_weight > 0.0:
            self.disc_tokens = 4 # extra as register tokens
            self.disc_model = TiTokEncoder( # same arch as tokenizer encoder
                model_size=model_d.model_size,
                patch_size=model_d.patch_size,
                in_channels=3,
                out_channels=1, # more stable to use more channels?
            ).apply(init_weights)

            if config.training.main.torch_compile:
                self.disc_model = torch.compile(self.disc_model)

        self.gp_weight = loss_d.gp_weight
        self.gp_noise = loss_d.gp_noise
        self.centering_weight = loss_d.centering_weight
        self.total_steps = config.training.main.max_steps
        

    def perceptual_preprocess(self, target, recon, resize_prob=0.25):
        target_out = []
        recon_out = []
        sample_size = self.config.tokenizer.losses.perceptual_sampling_size
        perceptual_samples = self.config.tokenizer.losses.perceptual_samples_per_step

        if perceptual_samples == -1:
            perceptual_samples = len(target)

        for i, (trg_frame, rec_frame) in enumerate(sorted(zip(target, recon), key=lambda k: random.random())): # randomize order | list() before sorted?
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

            if i >= perceptual_samples:
                break 

        target = torch.stack(target_out, dim=0).contiguous() # now FCHW
        recon = torch.stack(recon_out, dim=0).contiguous()
        return recon, target
    
    
    def disc_wrapper(self, x):
        B = len(x)
        # logits = self.disc_model(x, token_counts=[self.disc_tokens]*B).view(B, -1).mean(-1)
        token_counts = torch.tensor([self.disc_tokens], device=x[0].device, dtype=torch.int32).repeat(B)
        logits = self.disc_model(x, token_counts).view(B, -1).mean(-1)
        return logits
        

    def forward(self, target, recon, disc_forward=False):
        if disc_forward:
            return self._forward_discriminator(target, recon)
        else:
            return self._forward_generator(target, recon)
        

    def _forward_generator(self, target, recon):
        # target and recon are now lists of CHW tensors.
        loss_dict = {}

        target = [i.contiguous() for i in target]
        recon = [i.contiguous() for i in recon]

        recon_loss = torch.stack([l1(x, y).mean() for x, y in zip(target, recon)]) # .mean() # not [B]
        loss_dict['recon_loss'] = recon_loss

        perceptual_loss = 0.0
        gram_loss = 0.0
        if self.perceptual_weight > 0.0 or self.gram_weight > 0.0:
            target_frames = []
            recon_frames = []
            for trg_vid, rec_vid in zip(target, recon):
                target_frames += trg_vid.unbind(1) # unbind T dim
                recon_frames += rec_vid.unbind(1)

            target_frames, recon_frames = self.perceptual_preprocess(target_frames, recon_frames)
            perceptual_loss, gram_loss = [l.mean() for l in self.perceptual_model(recon_frames, target_frames)]

            if self.perceptual_weight > 0.0:
                loss_dict['perceptual_loss'] = perceptual_loss

            if self.gram_weight > 0.0:
                loss_dict['gram_loss'] = gram_loss


        g_loss = 0.0
        if self.disc_weight > 0.0:
            target = [i.detach().contiguous() for i in target]

            ############################
            for param in self.disc_model.parameters():
                param.requires_grad = False
            ############################

            logits_real = self.disc_wrapper(target)
            logits_fake = self.disc_wrapper(recon)
            logits_relative = logits_fake - logits_real
            g_loss = F.softplus(-logits_relative) # .mean()
            loss_dict['g_loss'] = g_loss

        total_loss = (
            recon_loss
            + (self.perceptual_weight * perceptual_loss)
            + (self.gram_weight * gram_loss)
            + (self.disc_weight * g_loss)
        ).mean()
        loss_dict['total_loss'] = total_loss

        return total_loss, {'gen/'+k:v.clone().mean().detach() for k,v in loss_dict.items()}
    
    
    def _forward_discriminator(self, target, recon):
        loss_dict = {}
        
        target = [i.detach().requires_grad_(True).contiguous() for i in target]
        recon = [i.detach().requires_grad_(True).contiguous() for i in recon]

        ############################
        for param in self.disc_model.parameters():
            param.requires_grad = True
        ############################

        logits_real = self.disc_wrapper(target)
        logits_fake = self.disc_wrapper(recon)
        logits_relative = logits_real - logits_fake
        d_loss = F.softplus(-logits_relative) # .mean()

        loss_dict['d_loss'] = d_loss
        loss_dict['logits_relative'] = logits_relative


        # https://www.arxiv.org/pdf/2509.24935
        gradient_penalty = 0.0
        if self.gp_weight > 0.0:
            noise = [torch.randn_like(x) * self.gp_noise for x in target] # diff noise per sample? averages out?
            logits_real_noised = self.disc_wrapper([x + y for x, y in zip(target, noise)])
            logits_fake_noised = self.disc_wrapper([x + y for x, y in zip(recon, noise)])

            r1_penalty = (logits_real - logits_real_noised)**2
            r2_penalty = (logits_fake - logits_fake_noised)**2

            loss_dict['r1_penalty'] = r1_penalty
            loss_dict['r2_penalty'] = r2_penalty
            gradient_penalty = r1_penalty + r2_penalty


        centering_loss = 0.0
        if self.centering_weight > 0.0:
            centering_loss = ((logits_real + logits_fake) ** 2) / 2
            loss_dict['centering_loss'] = centering_loss


        total_loss = (
            d_loss
            + (self.gp_weight / self.gp_noise**2 * gradient_penalty)
            + (self.centering_weight * centering_loss)
        ).mean()
        loss_dict['total_loss'] = total_loss

        return total_loss, {'disc/'+k:v.clone().mean().detach() for k,v in loss_dict.items()}