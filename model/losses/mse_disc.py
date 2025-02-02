import torch
import torch.nn as nn
import torch.nn.functional as F

from model.discriminator.titok_discriminator import TiTokDiscriminator
from einops import rearrange

import numpy as np

def zero_centered_grad_penalty(samples, critics):
    """Modified from https://github.com/brownvc/R3GAN"""
    gradient, = torch.autograd.grad(outputs=critics.sum(), inputs=samples, create_graph=True)
    return gradient.square().sum([1, 2, 3, 4]) # BCHW -> BCTHW? + last dim for video inputs

def lecam_reg(real_pred, fake_pred, lecam_ema):
    reg = torch.mean(F.relu(real_pred - lecam_ema.logits_fake_ema).pow(2)) + torch.mean(
        F.relu(lecam_ema.logits_real_ema - fake_pred).pow(2)
    )
    return reg


class LeCAM_EMA(object):
    # https://github.com/TencentARC/SEED-Voken/blob/main/src/Open_MAGVIT2/modules/losses/vqperceptual.py
    def __init__(self, init=0.0, decay=0.999):
        self.logits_real_ema = init
        self.logits_fake_ema = init
        self.decay = decay

    def update(self, logits_real, logits_fake):
        self.logits_real_ema = self.logits_real_ema * self.decay + torch.mean(logits_real).item() * (1 - self.decay)
        self.logits_fake_ema = self.logits_fake_ema * self.decay + torch.mean(logits_fake).item() * (1 - self.decay)

    
class ReconstructionLoss(nn.Module):    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        cd = config.model.disc
        self.use_disc = cd.use_disc

        if self.use_disc:
            cd = config.model.disc
            self.disc_start = cd.disc_start
            self.disc_factor = cd.disc_factor
            self.disc_weight = cd.disc_weight
            self.disc_model = TiTokDiscriminator( # 2d
                in_channels=config.model.vae.latent_channels,
                num_layers=cd.model_layers,
                d_model=cd.model_dim,
                num_heads=cd.model_heads,
                spatial_patch_size=cd.spatial_patch_size,
                in_spatial_size=(config.dataset.resolution // config.model.vae.spatial_compression),
            )

            self.lecam_weight = cd.lecam_weight
            if self.lecam_weight > 0:
                self.lecam_ema = LeCAM_EMA()

            self.base_gamma = cd.base_gamma
            self.final_gamma = cd.final_gamma

        self.total_steps = config.training.max_steps
        self.mse_loss = nn.MSELoss(reduction="mean")

    def cosine_decay(self, global_step):
        decay = 0.5 * (1 + np.cos(np.pi * global_step / self.total_steps))
        cur_value = self.base_gamma + (1 - decay) * (self.final_gamma - self.base_gamma)
        return float(np.where(global_step > self.total_steps, self.final_gamma, cur_value))

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.disc_weight
        return d_weight

    def forward(self, target, recon, global_step, last_layer=None, disc_forward=False):
        if disc_forward:
            return self._forward_discriminator(target, recon, global_step)
        else:
            return self._forward_generator(target, recon, global_step, last_layer)

    def _forward_generator(self, target, recon, global_step, last_layer=None):
        target = target.detach().contiguous()
        recon = recon.contiguous()

        recon_loss = self.mse_loss(rearrange(recon, "b c t h w -> (b t) c h w"), rearrange(target, "b c t h w -> (b t) c h w"))

        # adversarial loss
        if self.use_disc and global_step > self.disc_start:
            g_loss = torch.zeros((), device=target.device)
            logits_relative = torch.zeros((), device=target.device)
            disc_factor = self.disc_factor
            d_weight = self.disc_weight

            if global_step >= self.disc_start and self.disc_weight > 0.0:
                for param in self.disc_model.parameters():
                    param.requires_grad = False

                # logits_real, logits_fake = self.disc_model(target, recon)
                logits_real = self.disc_model(rearrange(target, "b c t h w -> (b t) c h w"))
                logits_fake = self.disc_model(rearrange(recon, "b c t h w -> (b t) c h w"))

                logits_relative = logits_fake - logits_real
                g_loss = F.softplus(-logits_relative).mean()

                # adaptive disc weight
                # try:
                #     d_weight = self.calculate_adaptive_weight(recon_loss, g_loss, last_layer=last_layer)
                # #    loss_dict["d_weight"] = d_weight.clone().detach()
                # except RuntimeError:
                #     assert not self.training
                #     d_weight = torch.tensor(0.0)
                # ######################

                total_loss = recon_loss + d_weight * disc_factor * g_loss

                loss_dict = {
                    "total_loss": total_loss.detach(),
                    "recon_loss": recon_loss.detach(),
                    "logits_relative": logits_relative.detach().mean(),
                    "gan_loss": g_loss.detach(),
                }
        else:
            total_loss = recon_loss
            loss_dict = {
                "total_loss": total_loss.detach(),
                "recon_loss": recon_loss.detach(),
            }


        return total_loss, {'train/'+k:v for k,v in loss_dict.items()}
    
    def _forward_discriminator(self, target, recon, global_step):

        assert self.use_disc, f"Discriminator not enabled."

        for param in self.disc_model.parameters():
            param.requires_grad = True
    
        target = target.detach().requires_grad_(True).contiguous()
        recon = recon.detach().requires_grad_(True).contiguous()

        # logits_real, logits_fake = self.disc_model(target, recon)
        logits_real = self.disc_model(rearrange(target, "b c t h w -> (b t) c h w"))
        logits_fake = self.disc_model(rearrange(recon, "b c t h w -> (b t) c h w"))

        r1_penalty = zero_centered_grad_penalty(target, logits_real)
        r2_penalty = zero_centered_grad_penalty(recon, logits_fake)

        logits_relative = logits_real - logits_fake
        rel_loss = F.softplus(-logits_relative)

        gamma = self.cosine_decay(global_step)

        adv_loss = (rel_loss + (gamma / 2) * (r1_penalty + r2_penalty).mean()).mean()

        
        if self.lecam_weight > 0:
            self.lecam_ema.update(logits_real, logits_fake)
            lecam_loss = lecam_reg(logits_real, logits_fake, self.lecam_ema)
            disc_loss = self.disc_factor * (lecam_loss * self.lecam_weight + adv_loss)
        else:
           disc_loss = self.disc_factor * adv_loss
        
        loss_dict = {
            "disc_loss": disc_loss.detach(),
            "logits_real": logits_real.detach().mean(),
            "logits_fake": logits_fake.detach().mean(),
            "r1_penalty": r1_penalty.detach().mean(),
            "r2_penalty": r2_penalty.detach().mean(),
            # "penalty_gamma": gamma,
        }

        if self.lecam_weight > 0:
            loss_dict.update({"lecam_loss": lecam_loss.detach()})
            
        return disc_loss, {'train/'+k:v for k,v in loss_dict.items()}