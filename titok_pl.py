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
import math

from dataset.video_dataset import WebdatasetVideoDataModule
from model.titok import TiTok
from model.losses.loss_module import ReconstructionLoss
from train_utils.lr_schedulers import get_scheduler
from train_utils.codebook_logging import CodebookLogger

    
class TitokTrainer(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_disc = config.model.disc.get('use_disc', False)
        self.clip_grads = config.training.get('max_grad_norm', False)
        self.model = TiTok(config)

        self.eval_metrics = MetricCollection(
            {
                "psnr": PeakSignalNoiseRatio(),
                "ssim": StructuralSimilarityIndexMeasure(),
                "lpips": LearnedPerceptualImagePatchSimilarity(net_type='vgg').eval(),
            },
            prefix="eval/",
        )

        self.eval_metrics['lpips'].requires_grad_(False)

        if config.training.torch_compile:
            # torch._dynamo.config.compiled_autograd = True
            self.model = torch.compile(self.model)
            self.eval_metrics['lpips'] = torch.compile(self.eval_metrics['lpips'])
            if self.use_disc:
                self.loss_module.disc_model = torch.compile(self.loss_module.disc_model)

        self.loss_module = ReconstructionLoss(config, self.eval_metrics['lpips'])

        self.seen_recon = 0

        if config.training.log_codebook:
            self.codebook_logger = CodebookLogger(codebook_size=math.prod(config.model.titok.fsq_levels))

        self.automatic_optimization = False
        self.strict_loading = False # to allow loading from lpips-less checkpoint

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

        ############################
        self.toggle_optimizer(opt_g)

        x, results_dict = self.model(orig)
        loss, loss_dict = self.loss_module(orig, x, self.global_step, last_layer=self.model.decoder.conv_out[-1].weight, results_dict=results_dict) # normally -1

        opt_g.zero_grad(set_to_none=True)
        self.manual_backward(loss)
        if self.clip_grads:
            self.clip_gradients(opt_g, gradient_clip_val=config.training.max_grad_norm)
        if self.global_step % self.config.training.val_step_interval == 0:
            self.log_dict(grad_norm(self.model, norm_type=2))
        opt_g.step()
        sched_g.step()
        self.untoggle_optimizer(opt_g)
        ############################

        if self.use_disc and self.global_step >= self.config.model.disc.disc_start and self.global_step % self.config.model.disc.every_n == 0:
            self.toggle_optimizer(opt_d)

            d_loss, d_loss_dict = self.loss_module(orig, x, self.global_step, disc_forward=True)
            loss_dict.update(d_loss_dict)

            opt_d.zero_grad(set_to_none=True)
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
            self.codebook_logger(results_dict['codes'])

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            orig = batch['video']
            recon, results_dict = self.model(orig)
            recon = recon.clamp(-1, 1)

            self.eval_metrics.update(rearrange(recon, "b c t h w -> (b t) c h w"), rearrange(orig, "b c t h w -> (b t) c h w"))

        for x, y in zip(recon, orig):
            if self.seen_recon < self.config.training.eval_recon_log_num: 
                merged_video = torch.cat((y, x), dim=-1).permute(1, 0, 2, 3).cpu().float().numpy() # tch(W) concat
                merged_video = ((merged_video + 1) / 2 * 255).astype(np.uint8)
                self.seen_recon += 1
                self.logger.log_video(key=f"Video recon {self.seen_recon}", videos=[merged_video], step=self.global_step, fps=[self.config.dataset.frames_per_second])


    def on_validation_epoch_end(self):
        self.logger.log_metrics(self.eval_metrics.compute(), step=self.global_step)
        self.eval_metrics.reset()
        self.seen_recon = 0

        if self.config.training.log_codebook and self.codebook_logger.is_score_ready():
            self.logger.log_metrics(self.codebook_logger.get_scores(), step=self.global_step)

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
        # Don't save lpips
        return {k: v for k, v in super().state_dict().items() if 'eval_metrics' not in k}
    
    def on_load_checkpoint(self, checkpoint):
        if self.config.logging.discard_disc_on_resume and self.config.model.disc.use_disc: # keeps everything but the disc and disc optim. For use where changing disc size.
            opt_conf_d = self.config.optimizer.disc
            opt_d = optim.AdamW(
                self.loss_module.disc_model.parameters(),
                lr=opt_conf_d.learning_rate,
                weight_decay=opt_conf_d.weight_decay,
                betas=[opt_conf_d.beta1, opt_conf_d.beta2])

            orig_sd = checkpoint['state_dict']
            new_sd = {}

            for k, v in orig_sd.items():
                if not k.startswith('loss_module.disc_model'):
                    new_sd[k] = v

            checkpoint['state_dict'] = new_sd
            checkpoint['optimizer_states'][1] = opt_d.state_dict() # fresh optim

            # don't clear disc LR sched?



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

    checkpoint_callback = ModelCheckpoint(
        dirpath=config.logging.save_path,
        every_n_train_steps=config.logging.save_step_interval,
        save_top_k=config.logging.keep_prior_checkpoints,
        monitor='step', mode='max', # allow saving of N number of most recent checkpoints by highest step count.
    )
    
    wandb_logger = WandbLogger(
        name=config.logging.run_name,
        project=config.logging.project,
    )

    dataloaders = WebdatasetVideoDataModule(config)

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
        limit_val_batches=config.training.eval_sample_size//config.training.eval_batch_size,
    )

    model_trainer = TitokTrainer(config)

    if init_path:
        orig_sd = torch.load(config.logging.init_from_checkpoint, map_location="cpu", weights_only=False)['state_dict']
        model_trainer.load_state_dict(orig_sd, strict=False)

    trainer.fit(
        model_trainer,
        train_dataloaders=dataloaders.train_dataloader(),
        val_dataloaders=dataloaders.eval_dataloader(),
        ckpt_path=resume_path if resume_path else None,
    )