"""Training utils for TiTok.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License.
"""
import os
from pathlib import Path

from torch.utils.data import DataLoader
from dataset.encoded import EncodedDataset

from base_tokenizers.tokenizer_wrapper import WFVAEWrapper, CogvideoXVAEWrapper

import torch
from omegaconf import OmegaConf
from torch.optim import AdamW
from utils.lr_schedulers import get_scheduler
from model.losses.mse import ReconstructionLoss
from model.titok import TiTok

from torchinfo import summary

def get_config():
    """Reads configs from a yaml file and terminal."""
    cli_conf = OmegaConf.from_cli()

    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    return conf


class AverageMeter(object):
    """Computes and stores the average and current value.
    
    This class is borrowed from
    https://github.com/pytorch/examples/blob/main/imagenet/main.py#L423
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_pretrained_vae(config, accelerator):
    accelerator.print("Loading VAE.")

    if config.model.pretrained_vae.vae_dtype == 'bf16':
        torch_dtype = torch.bfloat16
    elif config.model.pretrained_vae.vae_dtype == 'fp16':
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    if config.model.pretrained_vae.type == 'wfvae':
        vae = WFVAEWrapper(model_path=config.model.pretrained_vae.path, dtype=torch_dtype, embed_dim=config.model.pretrained_vae.latent_channels)
    elif config.model.pretrained_vae.type == 'cogvideox':
        vae = CogvideoXVAEWrapper(model_path=config.model.pretrained_vae.path, dtype=torch_dtype, embed_dim=config.model.pretrained_vae.latent_channels)

    return vae


def create_model_and_loss_module(config, accelerator):
    """Creates TiTok model and loss module."""
    accelerator.print("Creating model and loss module.")

    model = TiTok(config)

    if config.experiment.get("init_weight", ""):
        # If loading a pretrained weight - use from_pretrained for safetensors?
        model_weight = torch.load(config.experiment.init_weight, map_location="cpu")
        msg = model.load_state_dict(model_weight['model'], strict=False)
        accelerator.print(f"loading weight from {config.experiment.init_weight}, msg: {msg}")

    # Create loss module
    loss_module = ReconstructionLoss(config=config)

    temporal_size = config.video_dataset.params.num_frames // config.model.pretrained_vae.temporal_compression
    spatial_size = config.video_dataset.params.resolution // config.model.pretrained_vae.spatial_compression

    input_size = (1, config.model.pretrained_vae.latent_channels, temporal_size, spatial_size, spatial_size)
    model_summary_str = summary(model, input_size=input_size, depth=5, col_names=("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds"))
    accelerator.print(model_summary_str)

    return model, loss_module


def create_optimizer(config, accelerator, model):
    """Creates optimizer for TiTok."""
    accelerator.print("Creating optimizers.")
    optimizer_config = config.optimizer.params
    learning_rate = optimizer_config.learning_rate

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer_cls = AdamW
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    # Exclude terms we may not want to apply weight decay.
    exclude = (lambda n, p: p.ndim < 2 or "ln" in n or "bias" in n or 'latent_tokens' in n 
               or 'mask_token' in n or 'embedding' in n or 'norm' in n or 'gamma' in n)
    include = lambda n, p: not exclude(n, p)
    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
    

    optimizer = optimizer_cls(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": optimizer_config.weight_decay},
        ],
        lr=learning_rate,
        betas=(optimizer_config.beta1, optimizer_config.beta2)
    )

    return optimizer


def create_lr_scheduler(config, accelerator, optimizer):
    """Creates learning rate scheduler for TiTok and discrminator."""
    accelerator.print("Creating lr_schedulers.")
    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.experiment.max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
        base_lr=config.lr_scheduler.params.learning_rate,
        end_lr=config.lr_scheduler.params.end_lr,
    )
    return lr_scheduler


def create_video_dataloader(config, accelerator):
    """Creates data loader for training and testing."""
    accelerator.print("Creating dataloaders.")
    batch_size = config.training.per_gpu_batch_size
    video_dataset_config = config.video_dataset.params

    train_dataset = EncodedDataset( # for train and codebook eval
        video_dataset_config.train_dataset,
    )

    eval_dataset = EncodedDataset( # for ssim/psnr and recon
        video_dataset_config.eval_dataset,
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=config.video_dataset.params.dataset_workers, persistent_workers=True
    )

    codebook_dataloader = DataLoader(
        train_dataset, batch_size=config.training.codebook_eval_batch_size, shuffle=True, pin_memory=True, num_workers=config.video_dataset.params.dataset_workers, persistent_workers=True
    )

    recon_dataloader = DataLoader(
        eval_dataset, batch_size=config.training.recon_batch_size, shuffle=False, pin_memory=True
    )
    
    return train_dataloader, codebook_dataloader, recon_dataloader


def auto_resume(config, accelerator, num_update_steps_per_epoch, strict=True):
    """Auto resuming the training."""
    global_step = 0
    first_epoch = 0
    # If resuming training.
    if config.experiment.resume:
        dirs = [f for f in os.scandir(config.experiment.output_dir) if f.is_dir() and 'checkpoint' in os.path.basename(f)]
        dirs.sort(key=os.path.getctime)
        if len(dirs) >= 1:
            latest_path = dirs[-1]
            global_step = load_checkpoint(latest_path, accelerator, strict=strict)
            first_epoch = global_step // num_update_steps_per_epoch
        else:
            accelerator.print("Training from scratch.")
    return global_step, first_epoch

def save_checkpoint(config, accelerator, global_step):
    os.makedirs(config.experiment.output_dir, exist_ok=True)
    save_path = Path(config.experiment.output_dir) / f"checkpoint-{global_step}"
    accelerator.save_state(output_dir=save_path)
    accelerator.print(f"Saved state to {save_path}")

def load_checkpoint(checkpoint_path, accelerator, strict=True):
    accelerator.load_state(checkpoint_path, strict=strict)
    global_step = int(os.path.basename(checkpoint_path).split('-')[-1])
    accelerator.print(f"Loading checkpoint from: {checkpoint_path} | Resuming from step: {global_step}")
    return global_step

def log_grad_norm(model):
    grad_dict = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            grad_dict["grad_norm/" + name] = grad_norm
    return grad_dict




