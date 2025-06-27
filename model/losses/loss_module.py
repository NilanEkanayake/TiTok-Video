import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np

from model.discriminator.vit_disc import ViTDiscriminator
from model.discriminator.n_layer import NLayerDiscriminatorSpectral, NLayerDiscriminatorSpectral3D, weights_init

from model.metrics.lpips import LPIPS
from model.metrics.cqvqm import CGVQM

def zero_centered_grad_penalty(samples, critics):
    """Modified from https://github.com/brownvc/R3GAN"""
    gradient, = torch.autograd.grad(outputs=critics.sum(), inputs=samples, create_graph=True)
    return gradient.square().sum([1, 2, 3, 4])

def l1(x, y):
    return torch.abs(x - y)
    
class ReconstructionLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.perceptual_weight = config.losses.recon.perceptual_weight
        if self.perceptual_weight > 0.0:
            self.perceptual_model = LPIPS().eval()
            # self.perceptual_model = CGVQM().eval()
            for param in self.perceptual_model.parameters():
                param.requires_grad = False

        self.dwt_weight = config.losses.recon.dwt_weight

        disc_conf = config.model.disc
        ds_conf = config.dataset

        self.use_disc = config.losses.disc.use_disc
        self.disc_weight = config.losses.disc.disc_weight
        self.disc_start = config.losses.disc.disc_start

        if self.use_disc:
            self.disc_model = ViTDiscriminator(
                model_size=disc_conf.model_size,
                in_grid=(ds_conf.num_frames, ds_conf.resolution, ds_conf.resolution),
                in_channels=3,
                patch_size=(disc_conf.temporal_patch_size, disc_conf.spatial_patch_size, disc_conf.spatial_patch_size),
                out_tokens=1,
            )

            # self.disc_model = NLayerDiscriminatorSpectral3D(input_nc=3, output_nc=1).apply(weights_init)

            if config.training.main.torch_compile:
                self.disc_model = torch.compile(self.disc_model)

            self.base_gamma = config.losses.disc.base_gamma
            self.final_gamma = config.losses.disc.final_gamma

        self.total_steps = config.training.main.max_steps
    
    def cosine_decay(self, global_step):
        decay = 0.5 * (1 + np.cos(np.pi * global_step / self.total_steps))
        cur_value = self.base_gamma + (1 - decay) * (self.final_gamma - self.base_gamma)
        return float(np.where(global_step > self.total_steps, self.final_gamma, cur_value))

    def forward(self, target, recon, global_step, results_dict=None, disc_forward=False):
        if disc_forward:
            return self._forward_discriminator(target, recon, global_step)
        else:
            return self._forward_generator(target, recon, global_step, results_dict)

    def _forward_generator(self, target, recon, global_step, results_dict=None):
        loss_dict = {}
        target = target.contiguous()
        recon = recon.contiguous()

        B, C, T, H, W = target.shape

        recon_loss = l1(rearrange(target, "b c t h w -> (b t) c h w"), rearrange(recon, "b c t h w -> (b t) c h w")).view(B, -1).mean(1) # [B]
        loss_dict['recon_loss'] = recon_loss

        perceptual_loss = 0.0
        if self.perceptual_weight > 0.0:
            num_subsample = self.config.losses.recon.perceptual_subsample
            if num_subsample != -1:
                batch_indices = torch.arange(B, device=target.device).repeat_interleave(num_subsample)
                random_frames = torch.randint(0, T, (B * num_subsample,), device=target.device)

                recon_sampled = recon[batch_indices, :, random_frames, :, :]
                target_sampled = target[batch_indices, :, random_frames, :, :]
                perceptual_loss = self.perceptual_model(recon_sampled.clamp(-1, 1).contiguous(), target_sampled.contiguous()).view(B, -1).mean(1) # [B]
            else:
                rec_frames = rearrange(recon.clamp(-1, 1), 'b c t h w -> (b t) c h w').contiguous()
                trg_frames = rearrange(target, 'b c t h w -> (b t) c h w').contiguous()
                perceptual_loss = self.perceptual_model(rec_frames, trg_frames).view(B, -1).mean(1) # [B]

            # perceptual_loss = self.perceptual_model(recon, target).mean()
            loss_dict['perceptual_loss'] = perceptual_loss

        # dwt
        dwt_loss = 0.0
        if self.dwt_weight > 0.0:
            target_dwt = results_dict['target_dwt'].contiguous()
            recon_dwt = results_dict['recon_dwt'].contiguous()
            recon_loss_low = l1(recon_dwt[:, :3], target_dwt[:, :3]).view(B, -1).mean(1) * 0.5 # [B]
            recon_loss_high = l1(recon_dwt[:, 3:], target_dwt[:, 3:]).view(B, -1).mean(1)
            dwt_loss = recon_loss_low + recon_loss_high
            loss_dict['dwt_loss'] = dwt_loss

        # adversarial loss
        d_weight = self.disc_weight
        d_weight_warm = min(1.0, ((global_step - self.disc_start) / self.config.losses.disc.disc_weight_warmup_steps))
        g_loss = 0.0
        if self.use_disc and global_step > self.disc_start:
            target = target.detach().contiguous()

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
            + (self.dwt_weight * dwt_loss)
        ).mean()

        loss_dict['total_loss'] = total_loss
            
        return total_loss, {'train/'+k:v.clone().mean().detach() for k,v in loss_dict.items()}
    
    def _forward_discriminator(self, target, recon, global_step):
        loss_dict = {}
        
        target = target.detach().requires_grad_(True).contiguous()
        recon = recon.detach().requires_grad_(True).contiguous()

        B, C, T, H, W = target.shape

        ############################
        for param in self.disc_model.parameters():
            param.requires_grad = True
        ############################

        logits_real = self.disc_model(target).view(B, -1).mean(1)
        logits_fake = self.disc_model(recon).view(B, -1).mean(1)

        # https://github.com/brownvc/R3GAN
        r1_penalty = zero_centered_grad_penalty(target, logits_real)
        r2_penalty = zero_centered_grad_penalty(recon, logits_fake)
 
        logits_relative = logits_real - logits_fake
        
        gamma = self.cosine_decay(global_step)
        disc_loss = (F.softplus(-logits_relative) + (gamma / 2) * (r1_penalty + r2_penalty)).mean()
        
        loss_dict.update({
            "disc_loss": disc_loss,
            "logits_real": logits_real,
            "logits_fake": logits_fake,
            "r1_penalty": r1_penalty,
            "r2_penalty": r2_penalty,
        })
            
        return disc_loss, {'train/'+k:v.clone().mean().detach() for k,v in loss_dict.items()}