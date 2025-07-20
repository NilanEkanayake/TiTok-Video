import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np

from model.discriminator.vit_disc import ViTDiscriminator
from model.metrics.lpips import LPIPS

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
        
        self.perceptual_weight = config.losses.recon.perceptual_weight
        if self.perceptual_weight > 0.0:
            self.perceptual_model = LPIPS().eval()

            for param in self.perceptual_model.parameters():
                param.requires_grad = False

        disc_conf = config.model.disc
        ds_conf = config.dataset

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
            target_padded = []
            recon_padded = []
            max_grid = self.config.dataset.max_grid
            for i in range(len(target)):
                padding = nn.ZeroPad2d((max_grid[2]-target[i].shape[3], 0, max_grid[1]-target[i].shape[2], 0)) # H and W padding is applied backwards
                target_padded.append(padding(target[i].transpose(0, 1))) # CTHW -> TCHW
                recon_padded.append(padding(recon[i].transpose(0, 1)))
            target_padded = torch.cat(target_padded, dim=0).contiguous() # now (BT)CHW
            recon_padded = torch.cat(recon_padded, dim=0).contiguous().clamp(-1, 1) # now (BT)CHW

            # since B and T are all mixed up now, just get total num to randomly sample across the entire (BT) dim. The next best would be a subsample percent of the frames in each sequence.
            num_subsample = self.config.losses.recon.perceptual_subsample_per_batch
            if num_subsample != -1:
                # sample num_subsample frames along dim 0 of target and recon (same indices for each)
                # then calculate the perceptual loss over than (num_subsample)CHW tensor.
                total_frames = target_padded.shape[0]
                if num_subsample < total_frames:
                    indices = torch.randperm(total_frames)[:num_subsample]
                    target_subsampled = target_padded[indices]
                    recon_subsampled = recon_padded[indices]
                    perceptual_loss = self.perceptual_model(recon_subsampled, target_subsampled).mean()
                else:
                    perceptual_loss = self.perceptual_model(recon_padded, target_padded).mean()
            else:
                perceptual_loss = self.perceptual_model(recon_padded, target_padded).mean() #.view(B, -1).mean(1) # [B]

            # perceptual_loss = self.perceptual_model(recon, target).mean()
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