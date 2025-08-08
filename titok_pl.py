import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities import grad_norm
from torch import optim, nn
import torch.nn.functional as F

from einops import rearrange
from omegaconf import OmegaConf
import numpy as np
import math

from dataset.video_dataset import WebdatasetVideoDataModule
from model.titok import TiTok
from model.losses.loss_module import ReconstructionLoss
from train_utils.lr_schedulers import get_scheduler
from train_utils.codebook_logging import CodebookLogger

from model.metrics.eval_metrics import EvalMetrics
import random

    
class TitokTrainer(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_disc = config.losses.disc.use_disc
        self.clip_grads = config.training.main.get('max_grad_norm', False)

        self.model = TiTok(config)
        self.eval_metrics = EvalMetrics(config)
        self.loss_module = ReconstructionLoss(config)

        if config.training.main.torch_compile:
            # torch._dynamo.config.compiled_autograd = True
            self.model = torch.compile(self.model)

        if config.training.eval.log_codebook:
            self.codebook_logger = CodebookLogger(codebook_size=math.prod(config.model.titok.fsq_levels))

        if config.losses.disc.use_disc and config.losses.disc.freeze_encoder:
            for param in self.model.encoder.parameters():
                param.requires_grad = False
        
        self.automatic_optimization = False
        self.strict_loading = False # to allow loading from lpips-less checkpoint


    def training_step(self, batch, batch_idx):
        orig = batch['video']
        token_counts = batch['token_count']

        if self.use_disc:
            opt_g, opt_d = self.optimizers()
            sched_g, sched_d = self.lr_schedulers()

            # Bugfix, see: https://github.com/Lightning-AI/pytorch-lightning/issues/17958
            opt_d._on_before_step = lambda : self.trainer.profiler.start("optimizer_step")
            opt_d._on_after_step = lambda : self.trainer.profiler.stop("optimizer_step")
        else:
            opt_g = self.optimizers()
            sched_g = self.lr_schedulers()

        ############################
        self.toggle_optimizer(opt_g)

        x, results_dict = self.model(orig, token_counts)

        loss, loss_dict = self.loss_module(
            target=orig,
            recon=x,
            global_step=self.global_step,
            results_dict=results_dict,
        )

        self.manual_backward(loss)
        if self.clip_grads:
            self.clip_gradients(opt_g, gradient_clip_val=self.config.training.main.max_grad_norm)
        if self.global_step % self.config.training.eval.eval_step_interval == 0:
            self.log_dict(grad_norm(self.model, norm_type=2))
        opt_g.step()
        sched_g.step()
        opt_g.zero_grad(set_to_none=True)
        self.untoggle_optimizer(opt_g)
        ############################

        if self.use_disc and self.global_step >= self.config.losses.disc.disc_start and self.global_step % self.config.losses.disc.every_n == 0:
            # with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                self.toggle_optimizer(opt_d)

                d_loss, d_loss_dict = self.loss_module(
                    target=orig,
                    recon=x,
                    global_step=self.global_step,
                    disc_forward=True
                )
                loss_dict.update(d_loss_dict)

                self.manual_backward(d_loss)
                if self.clip_grads:
                    self.clip_gradients(opt_d, gradient_clip_val=self.config.training.main.max_grad_norm)
                if self.global_step % self.config.training.eval.eval_step_interval == 0:
                    self.log_dict(grad_norm(self.loss_module, norm_type=2))
                opt_d.step()
                sched_d.step()
                opt_d.zero_grad(set_to_none=True)
                self.untoggle_optimizer(opt_d)

        self.log_dict(loss_dict, prog_bar=True)

        if self.config.training.eval.log_codebook: # small speed hit?
            self.codebook_logger([i.detach().cpu() for i in results_dict['indices']])

    def on_validation_epoch_start(self):
        # recon sampling from eval dataset
        num_recon = self.config.training.eval.log_recon_num
        if self.config.training.eval.random_recon:
            self.recon_indexes = torch.randperm(self.config.training.eval.num_eval)[:num_recon].tolist() # random sampling
        else:
            self.recon_indexes = list(range(num_recon)) # first num_recon

        self.seen_eval = 0
        self.seen_recon = 0

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            orig = batch['video']
            fps = batch['fps']
            token_counts = batch['token_count']

            recon, _ = self.model(orig, token_counts)
            self.eval_metrics.update(recon, orig)

        for x, y, f, t in zip(recon, orig, fps, token_counts): # list of CTHW, recon and orig sharing tensor shape
            if self.seen_eval in self.recon_indexes:
                merged_video = torch.cat((y, x.clamp(-1, 1)), dim=-1).permute(1, 0, 2, 3).cpu().float().numpy() # tch(W) concat
                merged_video = ((merged_video + 1) / 2 * 255).astype(np.uint8)
                self.seen_recon += 1
                self.logger.log_video(
                    key=f"Video recon {self.seen_recon}",
                    videos=[merged_video],
                    step=self.global_step,
                    fps=[f],
                    caption=[f"{t} tokens"],
                    format=['mp4']
                )
            self.seen_eval += 1

    def on_validation_epoch_end(self):
        self.logger.log_metrics(self.eval_metrics.compute(), step=self.global_step)
        self.eval_metrics.reset()

        if self.config.training.eval.log_codebook and self.codebook_logger.is_score_ready():
            self.logger.log_metrics(self.codebook_logger.get_scores(), step=self.global_step)

        if self.config.training.eval.clear_cache:
            torch.cuda.empty_cache()

    def forward(self, x):
        pass

    def configure_optimizers(self):
        opt_conf_g = self.config.optimizer.titok

        # ###
        # exclude = (lambda n, p: p.ndim < 2 or "ln" in n or "bias" in n or 'latent_tokens' in n 
        #     or 'mask_tokens' in n or 'norm' in n or 'gamma' in n or 'sigma' in n or 'positional_embedding' in n) # also include projections?
        # include = lambda n, p: not exclude(n, p)
        # named_parameters = list(self.model.named_parameters())
        # gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        # rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
        # ###

        opt_g = optim.AdamW(
            self.model.parameters(),
            weight_decay=opt_conf_g.weight_decay,
            # [
            #     {"params": gain_or_bias_params, "weight_decay": 0.},
            #     {"params": rest_params, "weight_decay": opt_conf_g.weight_decay},
            # ],
            lr=opt_conf_g.learning_rate, 
            betas=[opt_conf_g.beta1, opt_conf_g.beta2],
        )

        lr_g = get_scheduler(
            name='cosine',
            optimizer=opt_g,
            num_warmup_steps=opt_conf_g.warmup_steps,
            num_training_steps=self.config.training.main.max_steps,
            base_lr=opt_conf_g.learning_rate,
            end_lr=opt_conf_g.end_lr,
        )

        if self.config.losses.disc.use_disc:
            opt_conf_d = self.config.optimizer.disc
            opt_d = optim.AdamW(
                self.loss_module.disc_model.parameters(),
                lr=opt_conf_d.learning_rate,
                weight_decay=opt_conf_d.weight_decay,
                betas=[opt_conf_d.beta1, opt_conf_d.beta2]
            )

            lr_d = get_scheduler(
                name='cosine',
                optimizer=opt_d,
                num_warmup_steps=opt_conf_d.warmup_steps,
                num_training_steps=self.config.training.main.max_steps - self.config.losses.disc.disc_start, # assume global_step starting from 0
                base_lr=opt_conf_d.learning_rate,
                end_lr=opt_conf_d.end_lr,
            )

            return [opt_g, opt_d], [lr_g, lr_d]
        else:
            return [opt_g], [lr_g]
        
    def state_dict(self):
        # Don't save metrics
        return {k: v for k, v in super().state_dict().items() if 'eval_metrics' not in k and 'perceptual_model' not in k}



if __name__ == '__main__':
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    config = OmegaConf.merge(yaml_conf, cli_conf)

    L.seed_everything(config.training.main.seed)
    if config.training.main.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # torch.set_float32_matmul_precision("medium")

    resume_path = config.general.checkpoints.get('resume_from_checkpoint', False)
    init_path = config.general.checkpoints.get('init_from_checkpoint', False)

    assert not (resume_path and init_path), 'Only one of resume_from_checkpoint and init_from_checkpoint should be specified.'

    checkpoint_callback = ModelCheckpoint(
        dirpath=config.general.checkpoints.save_path,
        every_n_train_steps=config.general.checkpoints.save_interval,
        save_top_k=config.general.checkpoints.keep_prior,
        monitor='step', mode='max', # allow saving of N number of most recent checkpoints by highest step count.
    )
    
    wandb_logger = WandbLogger(
        name=config.general.wandb.run_name,
        project=config.general.wandb.project,
    )

    dataloaders = WebdatasetVideoDataModule(config)

    trainer = L.Trainer(
        devices=config.training.main.train_devices,
        accelerator=config.training.main.accelerator,
        precision=config.training.main.precision,
        max_steps=config.training.main.max_steps,
        logger=wandb_logger,
        check_val_every_n_epoch=None,
        val_check_interval=config.training.eval.eval_step_interval,
        log_every_n_steps=config.general.wandb.log_step_interval,
        callbacks=[checkpoint_callback],
    )

    model_trainer = TitokTrainer(config)

    if init_path:
        model_sd = torch.load(config.general.checkpoints.init_from_checkpoint, map_location="cpu", weights_only=False)['state_dict']
        model_trainer.load_state_dict(model_sd, strict=False)

    trainer.fit(
        model_trainer,
        train_dataloaders=dataloaders.train_dataloader(),
        val_dataloaders=dataloaders.eval_dataloader(),
        ckpt_path=resume_path if resume_path else None,
    )
