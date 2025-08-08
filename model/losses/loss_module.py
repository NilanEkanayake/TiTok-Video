import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np

from model.discriminator.vit_disc import ViTDiscriminator
from model.metrics.lpips import LPIPS
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode
import random

def zero_centered_grad_penalty(samples, critics):
    """Modified from https://github.com/brownvc/R3GAN"""
    gradients = []
    for sample, critic in zip(samples, critics):
        gradient, = torch.autograd.grad(outputs=critic.sum(), inputs=sample, create_graph=True)
        gradients.append(gradient.square().sum()) # .square() applies element-wise, should be fine here.
    return torch.stack(gradients, dim=0) # [B, 1] out?

def l1(x, y):
    return torch.abs(x - y)

    
class ReconstructionLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        disc_conf = config.model.disc
        ds_conf = config.dataset


        self.perceptual_weight = config.losses.recon.perceptual_weight
        if self.perceptual_weight > 0.0:
            self.perceptual_model = LPIPS().eval()
            for param in self.perceptual_model.parameters():
                param.requires_grad = False

        self.use_disc = config.losses.disc.use_disc
        self.disc_weight = config.losses.disc.disc_weight
        self.disc_start = config.losses.disc.disc_start

        if self.use_disc:
            self.disc_model = ViTDiscriminator(
                model_size=disc_conf.model_size,
                patch_size=disc_conf.patch_size,
                out_tokens=1,
                in_channels=3,
                out_channels=16, # more stable to use more channels?
            )

            if config.training.main.torch_compile:
                self.disc_model = torch.compile(self.disc_model)

            self.base_gamma = config.losses.disc.base_gamma
            self.final_gamma = config.losses.disc.final_gamma

        self.total_steps = config.training.main.max_steps
    
    def cosine_decay(self, global_step):
        decay = 0.5 * (1 + np.cos(np.pi * global_step / self.total_steps))
        cur_value = self.base_gamma + (1 - decay) * (self.final_gamma - self.base_gamma)
        return float(np.where(global_step > self.total_steps, self.final_gamma, cur_value))
    
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

            logits_real = self.disc_model(target).view(B, -1).mean(1) # [B]
            logits_fake = self.disc_model(recon).view(B, -1).mean(1)

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

        logits_real = self.disc_model(target).view(B, -1).mean(1)
        logits_fake = self.disc_model(recon).view(B, -1).mean(1)

        # disc model outputs normal dense tensor

        # # https://github.com/brownvc/R3GAN
        # r1_penalty = zero_centered_grad_penalty(target, logits_real)
        # r2_penalty = zero_centered_grad_penalty(recon, logits_fake)
 
        logits_relative = logits_real - logits_fake
        
        # gamma = self.cosine_decay(global_step)
        # disc_loss = (F.softplus(-logits_relative) + (gamma / 2) * (r1_penalty + r2_penalty)).mean()
        disc_loss = F.softplus(-logits_relative).mean()
        
        loss_dict.update({
            "disc_loss": disc_loss,
            "logits_real": logits_real,
            "logits_fake": logits_fake,
            # "r1_penalty": r1_penalty,
            # "r2_penalty": r2_penalty,
        })
            
        return disc_loss, {'train/'+k:v.clone().mean().detach() for k,v in loss_dict.items()}