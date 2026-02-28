import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities import grad_norm
from torch import optim, nn
import torch.nn.functional as F

from collections import OrderedDict
from omegaconf import OmegaConf
from copy import deepcopy
from einops import rearrange
import numpy as np
import math

from dataset.video_dataset import WebdatasetVideoDataModule
from model.titok import TiTok

from model.losses.loss_module import ReconstructionLoss # Rp r1/r2 gan

from train_utils.lr_schedulers import get_scheduler
from train_utils.codebook_logging import CodebookLogger
from model.metrics.eval_metrics import EvalMetrics
import random

    
class TitokTrainer(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_disc = config.tokenizer.losses.disc_weight > 0.0
        self.clip_grads = config.training.main.get('max_grad_norm', False)

        self.model = TiTok(config)
        self.eval_metrics = EvalMetrics(config)
        self.loss_module = ReconstructionLoss(config)

        if config.training.main.torch_compile:
            self.model = torch.compile(self.model)

        if config.training.eval.log_codebook:
            self.codebook_logger = CodebookLogger(codebook_size=math.prod(config.tokenizer.model.fsq_levels))
        
        self.automatic_optimization = False
        self.strict_loading = False # to allow loading from lpips-less checkpoint


    def training_step(self, batch, batch_idx):
        orig = batch['video']
        token_counts = batch['token_counts']

        ###
        if self.use_disc:
            opt_g, opt_d = self.optimizers()
            sched_g, sched_d = self.lr_schedulers()
            # Bugfix, see: https://github.com/Lightning-AI/pytorch-lightning/issues/17958
            opt_d._on_before_step = lambda : self.trainer.profiler.start("optimizer_step")
            opt_d._on_after_step = lambda : self.trainer.profiler.stop("optimizer_step")
        else:
            opt_g = self.optimizers()
            sched_g = self.lr_schedulers()
        ###

        ############################
        self.toggle_optimizer(opt_g)
        opt_g.zero_grad(set_to_none=True)

        x, results_dict = self.model(orig, token_counts)

        loss, loss_dict = self.loss_module(
            target=orig,
            recon=x,
        )

        self.manual_backward(loss)
        if self.clip_grads:
            self.clip_gradients(opt_g, gradient_clip_val=self.clip_grads)
        if self.global_step % self.config.training.eval.eval_step_interval == 0:
            self.log_dict(grad_norm(self.model, norm_type=2))

        opt_g.step()
        sched_g.step(self.global_step)
        self.untoggle_optimizer(opt_g)
        loss_dict['g_lr'] = torch.tensor(sched_g.get_last_lr())

        ############################

        if self.use_disc:
            self.toggle_optimizer(opt_d)
            opt_d.zero_grad(set_to_none=True)

            d_loss, d_loss_dict = self.loss_module(
                target=orig,
                recon=x,
                disc_forward=True,
            )
            loss_dict.update(d_loss_dict)

            self.manual_backward(d_loss)
            if self.clip_grads:
                self.clip_gradients(opt_d, gradient_clip_val=self.clip_grads)
            if self.global_step % self.config.training.eval.eval_step_interval == 0:
                self.log_dict(grad_norm(self.loss_module, norm_type=2))

            opt_d.step()
            sched_d.step(self.global_step)
            self.untoggle_optimizer(opt_d)
            loss_dict['d_lr'] = torch.tensor(sched_d.get_last_lr())
        ############################
        
        loss_dict = {'train/'+k:v.clone().mean().detach() for k,v in loss_dict.items()}
        self.log_dict(loss_dict, prog_bar=True)

        if self.config.training.eval.log_codebook: # small speed hit?
            self.codebook_logger(torch.split(results_dict['indices'].detach().cpu(), token_counts.tolist(), dim=0))


    def on_validation_epoch_start(self):
        # recon sampling from eval dataset
        num_recon = self.config.training.eval.log_recon_num
        if self.config.training.eval.random_recon:
            self.recon_indexes = torch.randperm(self.config.training.eval.eval_samples)[:num_recon].tolist() # random sampling
        else:
            self.recon_indexes = list(range(num_recon)) # first num_recon

        self.seen_eval = 0
        self.seen_recon = 0


    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            orig = batch['video']
            fps = batch['fps']
            token_counts = batch['token_counts']

            recon, _ = self.model(orig, token_counts)
            self.eval_metrics.update(recon, orig)

        for x, y, f, t in zip(recon, orig, fps, token_counts): # list of CTHW, recon and orig sharing shape
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
        opt_conf_g = self.config.optimizer
        lr = opt_conf_g.learning_rate
        elr = opt_conf_g.end_lr
        dlr = opt_conf_g.disc_lr_ratio
        wd = opt_conf_g.weight_decay

        b1 = opt_conf_g.beta1
        b2 = opt_conf_g.beta2

        warm_steps = opt_conf_g.warmup_steps
        max_steps = self.config.training.main.max_steps
        
        opt_g = optim.AdamW(
            self.model.parameters(),
            weight_decay=wd,
            lr=lr, 
            betas=[b1, b2],
        )

        lr_g = get_scheduler(
            name='cosine',
            optimizer=opt_g,
            num_warmup_steps=warm_steps,
            num_training_steps=max_steps,
            base_lr=lr,
            end_lr=elr,
        )

        opt_d = optim.AdamW(
            self.loss_module.disc_model.parameters(),
            weight_decay=wd,
            lr=lr*dlr, 
            betas=[b1, b2],
        )

        lr_d = get_scheduler(
            name='cosine',
            optimizer=opt_d,
            num_warmup_steps=warm_steps,
            num_training_steps=max_steps,
            base_lr=lr*dlr,
            end_lr=elr*dlr,
        )

        return [opt_g, opt_d], [lr_g, lr_d]

        
    def state_dict(self):
        # Don't save metrics
        return {k: v for k, v in super().state_dict().items() if 'eval_metrics' not in k and 'perceptual_model' not in k}
    

if __name__ == '__main__':
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    config = OmegaConf.merge(yaml_conf, cli_conf)
    ct = config.training

    L.seed_everything(ct.main.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_float32_matmul_precision("high")
    if ct.main.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    resume_path = config.general.checkpoints.get('resume_from_checkpoint', False)
    init_path = config.general.checkpoints.get('init_from_checkpoint', False)
    assert not (resume_path and init_path), 'Only one of resume_from_checkpoint and init_from_checkpoint should be specified.'

    checkpoint_callback = ModelCheckpoint(
        dirpath=config.general.checkpoints.save_path,
        every_n_train_steps=config.general.checkpoints.save_interval,
        save_top_k=config.general.checkpoints.keep_prior,
        monitor='step', mode='max', # allow saving of N number of most recent checkpoints by highest step count.
    )
    
    wandb_logger = WandbLogger(name=config.general.wandb.run_name, project=config.general.wandb.project)
    dataloaders = WebdatasetVideoDataModule(config)
    model_trainer = TitokTrainer(config)

    if init_path:
        model_sd = torch.load(config.general.checkpoints.init_from_checkpoint, map_location="cpu", weights_only=False)
        model_trainer.load_state_dict(model_sd['state_dict'], strict=False)
        # model_trainer.global_step = model_sd['global_step']

    trainer = L.Trainer(
        devices=ct.main.train_devices,
        accelerator=ct.main.accelerator,
        precision=ct.main.precision,
        max_steps=ct.main.max_steps,
        logger=wandb_logger,
        check_val_every_n_epoch=None,
        val_check_interval=ct.eval.eval_step_interval,
        log_every_n_steps=config.general.wandb.log_step_interval,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(
        model_trainer,
        train_dataloaders=dataloaders.train_dataloader(),
        val_dataloaders=dataloaders.eval_dataloader(),
        ckpt_path=resume_path if resume_path else None,
    )
