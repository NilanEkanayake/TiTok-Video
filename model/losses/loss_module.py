import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# from model.discriminator.titok_disc import TiTokDiscriminator
from model.discriminator.n_layer import NLayerDiscriminator, NLayerDiscriminator3D, weights_init
# from model.discriminator.n_layer_compare import NLayerDiscriminator, NLayerDiscriminator3D, weights_init

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
    def __init__(self, config, lpips):
        super().__init__()
        self.config = config
        
        cd = config.model.disc
        self.use_disc = cd.use_disc

        self.lpips = lpips
        self.lpips_weight = config.model.titok.lpips_weight

        if self.use_disc:
            cd = config.model.disc
            self.disc_start = cd.disc_start
            self.disc_factor = cd.disc_factor
            self.disc_weight = cd.disc_weight

            self.disc_model = NLayerDiscriminator3D(
                input_nc=3*2, n_layers=cd.disc_layers, ndf=cd.disc_filters, #, use_actnorm=False
            ).apply(weights_init)

            # self.disc_model = TiTokDiscriminator(in_channels=3*2)
            
            self.lecam_weight = cd.lecam_weight
            if self.lecam_weight > 0:
                self.lecam_ema = LeCAM_EMA()

        self.total_steps = config.training.max_steps
        self.mse_loss = nn.MSELoss(reduction="mean")
    
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

    def forward(self, target, recon, global_step, last_layer=None, results_dict=None, disc_forward=False):
        if disc_forward:
            return self._forward_discriminator(target, recon, global_step)
        else:
            return self._forward_generator(target, recon, global_step, last_layer, results_dict)

    def _forward_generator(self, target, recon, global_step, last_layer=None, results_dict=None):
        loss_dict = {}
        target = target.contiguous()
        recon = recon.contiguous()

        recon_loss = self.mse_loss(rearrange(recon, "b c t h w -> (b t) c h w"), rearrange(target, "b c t h w -> (b t) c h w"))
        loss_dict['recon_loss'] = recon_loss.clone().detach()

        lpips_loss = 0.0
        num_subsample = self.config.model.titok.lpips_subsample
        if self.lpips_weight > 0.0:
            if num_subsample != -1:
                B, C, T, H, W = target.shape
                batch_indices = torch.arange(B, device=target.device).repeat_interleave(num_subsample)
                random_frames = torch.randint(0, T, (B * num_subsample,), device=target.device)

                recon_sampled = recon[batch_indices, :, random_frames, :, :]
                target_sampled = target[batch_indices, :, random_frames, :, :]
                lpips_loss = self.lpips(recon_sampled.clamp(-1, 1), target_sampled)
            else:
                lpips_loss = self.lpips(rearrange(recon.clamp(-1, 1), "b c t h w -> (b t) c h w"), rearrange(target, "b c t h w -> (b t) c h w"))
            loss_dict['lpips_loss'] = lpips_loss.clone().detach()

        # adversarial loss
        disc_factor = 0.0
        d_weight = 0.0
        g_loss = 0.0
        if self.use_disc and global_step > self.disc_start:
            g_loss = torch.zeros((), device=target.device)
            logits_relative = torch.zeros((), device=target.device)
            disc_factor = self.disc_factor
            d_weight = self.disc_weight

            ############################
            for param in self.disc_model.parameters():
                param.requires_grad = False
            ############################

            # recon = rearrange(recon, "b c t h w -> (b t) c h w")
            # target = rearrange(target, "b c t h w -> (b t) c h w")

            logits_real = self.disc_model(torch.cat((target, recon), dim=1))
            logits_fake = self.disc_model(torch.cat((recon, target), dim=1))
            logits_relative = logits_fake - logits_real
            g_loss = F.softplus(-logits_relative).mean()

            # logits_relative = self.disc_model(target, recon)
            # g_loss = F.softplus(-logits_relative).mean()

            # adaptive disc weight
            if self.training and self.config.model.disc.adapt_disc_weight:
                d_weight = self.calculate_adaptive_weight(recon_loss, g_loss, last_layer=last_layer)
                loss_dict['d_weight'] = d_weight.clone().detach()

            ######################
            loss_dict['gan_loss'] = g_loss.clone().detach()
            loss_dict['logits_relative'] = logits_relative.detach().mean()


        total_loss = (
            recon_loss 
            + (d_weight * disc_factor * g_loss) 
            + (self.lpips_weight * lpips_loss) 
        )

        loss_dict['total_loss'] = total_loss.clone().detach()
            
        return total_loss, {'train/'+k:v for k,v in loss_dict.items()}
    
    def _forward_discriminator(self, target, recon, global_step):
        loss_dict = {}

        # recon = rearrange(recon, "b c t h w -> (b t) c h w")
        # target = rearrange(target, "b c t h w -> (b t) c h w")
        
        target = target.requires_grad_(True).contiguous()
        recon = recon.detach().requires_grad_(True).contiguous()
        # target = target.contiguous()
        # recon = recon.detach().contiguous()

        ############################
        for param in self.disc_model.parameters():
            param.requires_grad = True
        ############################

        # Recon GAN. See https://arxiv.org/abs/2501.00103 section 2.1.2
        logits_real = self.disc_model(torch.cat((target, recon), dim=1)) # channel concat for n-layer
        logits_fake = self.disc_model(torch.cat((recon, target), dim=1))

        # https://github.com/AilsaF/RS-GAN and https://github.com/brownvc/R3GAN 
        logits_relative = logits_real - logits_fake
        adv_loss = F.softplus(-logits_relative).mean()

        # logits_relative = self.disc_model(target, recon)
        # adv_loss = F.softplus(logits_relative).mean()

        if self.lecam_weight > 0:
            self.lecam_ema.update(logits_real, logits_fake)
            lecam_loss = lecam_reg(logits_real, logits_fake, self.lecam_ema)
            disc_loss = self.disc_factor * (lecam_loss * self.lecam_weight + adv_loss)
            loss_dict["lecam_loss"] = lecam_loss.clone().detach()
        else:
            disc_loss = self.disc_factor * adv_loss
        
        loss_dict.update({
            "disc_loss": disc_loss.detach(),
            "logits_real": logits_real.detach().mean(),
            "logits_fake": logits_fake.detach().mean(),
        })
            
        return disc_loss, {'train/'+k:v for k,v in loss_dict.items()}