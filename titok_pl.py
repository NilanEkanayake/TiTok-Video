import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities import grad_norm
from torch import optim, nn
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio, LearnedPerceptualImagePatchSimilarity, StructuralSimilarityIndexMeasure
from torchmetrics import MetricCollection

from einops import rearrange
from omegaconf import OmegaConf
import numpy as np

from dataset.encoded import EncodedDataset
from model.titok import TiTok
from model.losses.mse_disc import ReconstructionLoss
from base_tokenizers import load_vae
from torch.utils.data import DataLoader
from utils.lr_schedulers import get_scheduler

import math
from scipy.stats import entropy
    
class TitokTrainer(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_disc = config.model.disc.get('use_disc', False)
        self.clip_grads = config.training.get('max_grad_norm', False)
        self.model = TiTok(config)

        self.loss_module = ReconstructionLoss(config)

        if config.training.torch_compile:
            self.model = torch.compile(self.model)
            # if self.use_disc:
            #     self.loss_module.disc_model = torch.compile(self.loss_module.disc_model)

        self.vae = load_vae(vae_name=config.model.vae.type, model_path=config.model.vae.path, embed_dim=config.model.vae.latent_channels)

        self.eval_metrics = MetricCollection(
            {
                "psnr": PeakSignalNoiseRatio(),
                "ssim": StructuralSimilarityIndexMeasure(),
                "lpips": LearnedPerceptualImagePatchSimilarity(net_type='vgg').eval(),
            },
            prefix="eval/",
        )

        self.seen_recon = 0
        self.codebook_size = math.prod(config.model.titok.fsq_levels)
        self.code_frequencies = torch.zeros(self.codebook_size)
        self.codebook_counter = 0
        self.codebook_indices = []

        for param in self.eval_metrics['lpips'].parameters():
            param.requires_grad = False

        for param in self.vae.parameters():
            param.requires_grad = False

        self.automatic_optimization = False
        self.strict_loading = False # to allow loading from vae/lpips-less checkpoint

    def training_step(self, batch, batch_idx):
        orig = batch['video']

        if self.use_disc:
            opt_g, opt_d = self.optimizers()
            sched_g, sched_d = self.lr_schedulers()

            # Bugfix, see: https://github.com/Lightning-AI/pytorch-lightning/issues/17958
            opt_d._on_before_step = lambda : self.trainer.profiler.start("optimizer_step")
            opt_d._on_after_step = lambda : self.trainer.profiler.stop("optimizer_step")
        else:
            opt_g = self.optimizers()
            sched_g = self.lr_schedulers()

        self.toggle_optimizer(opt_g)
        opt_g.zero_grad(set_to_none=True)

        x, results_dict = self.model(orig)
        loss, loss_dict = self.loss_module(orig, x, self.global_step)

        self.manual_backward(loss)
        if self.clip_grads:
            self.clip_gradients(opt_g, gradient_clip_val=config.training.max_grad_norm)
        if self.global_step % self.config.training.val_step_interval == 0:
            self.log_dict(grad_norm(self.model, norm_type=2))
        opt_g.step()
        sched_g.step()
        self.untoggle_optimizer(opt_g)

        if self.use_disc and self.global_step >= self.config.model.disc.disc_start:
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False): # fix R* penalties not working with FA
                self.toggle_optimizer(opt_d)
                opt_d.zero_grad(set_to_none=True)

                d_loss, d_loss_dict = self.loss_module(orig, x, self.global_step, disc_forward=True)

                loss_dict.update(d_loss_dict)
                self.manual_backward(d_loss)
                if self.clip_grads:
                    self.clip_gradients(opt_d, gradient_clip_val=config.training.max_grad_norm)
                if self.global_step % self.config.training.val_step_interval == 0:
                    self.log_dict(grad_norm(self.loss_module, norm_type=2))
                opt_d.step()
                sched_d.step()
                self.untoggle_optimizer(opt_d)


        self.log_dict(loss_dict, prog_bar=True)

        if self.config.training.log_codebook: # small speed hit?
            for sample in results_dict['codes']:
                if len(self.codebook_indices) == self.codebook_size:
                    self.codebook_indices.pop(0)
                self.codebook_indices.append(sample)

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            orig = self.vae.decode(batch['video'])
            recon, results_dict = self.model(batch['video'])
            recon = self.vae.decode(recon) # [-1, 1]

        for x, y in zip(recon, orig):
            if self.seen_recon < self.config.training.eval_recon_log_num: 
                merged_video = torch.cat((y, x), dim=-1).permute(1, 0, 2, 3).cpu().float().numpy() # tch(W) concat
                merged_video = ((merged_video + 1) / 2 * 255).astype(np.uint8)
                self.seen_recon += 1
                self.logger.log_video(key=f"Video recon {self.seen_recon}", videos=[merged_video], step=self.global_step, fps=[self.config.dataset.frames_per_second])

        recon = (rearrange(recon, "b c t h w -> (b t) c h w"))
        orig = (rearrange(orig, "b c t h w -> (b t) c h w"))
        self.eval_metrics.update(recon, orig)

    def on_validation_epoch_end(self):
        self.logger.log_metrics(self.eval_metrics.compute(), step=self.global_step)
        self.eval_metrics.reset()
        self.seen_recon = 0

        if len(self.codebook_indices) == self.codebook_size: # might not log every eval
            code_frequencies = torch.zeros(self.codebook_size)
            for sample in self.codebook_indices:
                code_frequencies += torch.bincount(sample, minlength=self.codebook_size).cpu()

            freq_np = code_frequencies.float().numpy()
            codebook_dict = {
                'codebook/usage_percent': (code_frequencies.count_nonzero() / self.codebook_size) * 100,
                'codebook/entropy': entropy(freq_np / freq_np.sum())
            }

            self.logger.log_metrics(codebook_dict, step=self.global_step)
            self.codebook_indices = []

        if self.config.training.eval_clear_cache:
            torch.cuda.empty_cache()

    def forward(self, x):
        pass

    def configure_optimizers(self):
        opt_conf_g = self.config.optimizer.titok
        opt_g = optim.AdamW(
            self.model.parameters(),
            weight_decay=opt_conf_g.weight_decay,
            lr=opt_conf_g.learning_rate, 
            betas=[opt_conf_g.beta1, opt_conf_g.beta2],
            # fused=True,
        )
        lr_g = get_scheduler(
            name='cosine',
            optimizer=opt_g,
            num_warmup_steps=opt_conf_g.warmup_steps,
            num_training_steps=self.config.training.max_steps,
            base_lr=opt_conf_g.learning_rate,
            end_lr=opt_conf_g.end_lr,
        )

        if self.config.model.disc.use_disc:
            opt_conf_d = self.config.optimizer.disc
            opt_d = optim.AdamW(
                self.loss_module.disc_model.parameters(),
                lr=opt_conf_d.learning_rate,
                weight_decay=opt_conf_d.weight_decay,
                betas=[opt_conf_d.beta1, opt_conf_d.beta2])
            lr_d = get_scheduler(
                name='cosine',
                optimizer=opt_d,
                num_warmup_steps=opt_conf_d.warmup_steps,
                num_training_steps=self.config.training.max_steps - self.config.model.disc.disc_start, # assume global_step starting from 0
                base_lr=opt_conf_d.learning_rate,
                end_lr=opt_conf_d.end_lr,
            )
            return [opt_g, opt_d], [lr_g, lr_d]
        else:
            return [opt_g], [lr_g]
        
    def state_dict(self):
        # Only save the model
        return {k: v for k, v in super().state_dict().items() if not k.startswith('vae') and not k.startswith('eval_metrics')}



if __name__ == '__main__':
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    config = OmegaConf.merge(yaml_conf, cli_conf)

    L.seed_everything(config.training.seed)
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    torch.set_float32_matmul_precision("medium")

    resume_path = config.logging.get('resume_from_checkpoint', False)
    init_path = config.logging.get('init_from_checkpoint', False)

    assert not (resume_path and init_path), 'Only one of resume_from_checkpoint and init_from_checkpoint should be specified.'

    checkpoint_callback = ModelCheckpoint(dirpath=config.logging.save_path, every_n_train_steps=config.logging.save_step_interval, save_top_k=config.logging.keep_prior_checkpoints)
    
    wandb_logger = WandbLogger(
        name=config.logging.run_name,
        project=config.logging.project,
    )

    trainer = L.Trainer(
        devices=config.training.train_devices,
        accelerator=config.training.accelerator,
        precision=config.training.precision,
        max_steps=config.training.max_steps,
        logger=wandb_logger,
        check_val_every_n_epoch=None,
        val_check_interval=config.training.val_step_interval,
        log_every_n_steps=config.logging.logging_interval,
        callbacks=[checkpoint_callback],
    )

    
    model_trainer = TitokTrainer(config)

    if init_path: # discards everything but the model weights
        orig_sd = torch.load(config.logging.init_from_checkpoint, map_location="cpu", weights_only=False)['state_dict']
        model_sd = {}
        for k, v in orig_sd.items():
            model_sd[k] = v
        model_trainer.load_state_dict(model_sd, strict=False)

    train_dataset = EncodedDataset(config.dataset.train_dataset, shuffle=True, shuffle_buffer=512)
    train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            num_workers=config.dataset.workers,
            pin_memory=True,
            persistent_workers=True,
        )
    
    eval_dataset = EncodedDataset(config.dataset.eval_dataset, shuffle=False, override_length=config.training.eval_sample_size)
    eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=config.training.eval_batch_size,
            num_workers=1,
            pin_memory=True,
            persistent_workers=True,
        )

    trainer.fit(
        model_trainer,
        train_dataloaders=train_dataloader,
        val_dataloaders=eval_dataloader,
        ckpt_path=resume_path if resume_path else None,
    )