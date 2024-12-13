"""Training script for TiTok.

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

Reference:
    https://github.com/huggingface/open-muse
"""
import math
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf
from utils.logger import setup_logger

from utils.latent_distillation_pretrain import train_one_epoch
from utils.utils import (
    get_config, create_pretrained_vae, 
    create_model_and_loss_module,
    create_optimizer, create_lr_scheduler, create_video_dataloader,
    auto_resume, save_checkpoint
)

from accelerate import Accelerator
from accelerate.utils import set_seed


def main():

    config = get_config()
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    output_dir = config.experiment.output_dir
    os.makedirs(output_dir, exist_ok=True)
    config.experiment.logging_dir = os.path.join(output_dir, "logs")

    logger = setup_logger(name="TiTok", log_level="INFO", output_file=f"{output_dir}/log.txt")

    accelerator = Accelerator(log_with='wandb', mixed_precision=config.training.mixed_precision, project_dir=os.getcwd(), device_placement=True)


    accelerator.init_trackers(
        project_name=config.experiment.project, 
        init_kwargs={"wandb": {"name": config.experiment.name}}
    )

    config_path = Path(output_dir) / "config.yaml"
    accelerator.print(f"Saving config to {config_path}")
    OmegaConf.save(config, config_path)


    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed, device_specific=True)

    pretrained_vae = create_pretrained_vae(config, accelerator)
    model, loss_module = create_model_and_loss_module(config, accelerator)
    optimizer = create_optimizer(config, accelerator, model)
    lr_scheduler = create_lr_scheduler(config, accelerator, optimizer)

    ########################
    max_train_steps = config.experiment.max_train_steps
    total_batch_size_without_accum = config.training.per_gpu_batch_size
    num_batches = math.ceil(config.experiment.max_train_examples / total_batch_size_without_accum)
    num_train_epochs = math.ceil(max_train_steps / num_batches)
    ########################

    train_dataloader, codebook_dataloader, recon_dataloader = create_video_dataloader(config, accelerator)

    model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(model, optimizer, lr_scheduler, train_dataloader)


    if config.training.torch_compile: # works with accelerate?
        model = model.to(accelerator.device)
        model = torch.compile(model=model, mode=config.training.torch_compile_mode, backend=config.training.torch_compile_backend)

    # Start training.
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {max_train_steps}")
    logger.info(f"  Batch size = { config.training.per_gpu_batch_size}")

    global_step, first_epoch = auto_resume(config, accelerator, num_batches, strict=True)


    for current_epoch in range(first_epoch, num_train_epochs):
        accelerator.print(f"Epoch {current_epoch+1}/{num_train_epochs} started.")

        global_step = train_one_epoch(
            config, logger, accelerator,
            model, loss_module,
            optimizer,
            lr_scheduler,
            train_dataloader, codebook_dataloader, recon_dataloader,
            global_step,
            max_train_steps,
            pretrained_vae
        )
        
        # Stop training if max steps is reached.
        if global_step >= max_train_steps:
            accelerator.print(f"Finishing training: Global step is >= Max train steps: {global_step} >= {max_train_steps}")
            break

    accelerator.wait_for_everyone()

    # Save checkpoint at the end of training.
    save_checkpoint(config, accelerator, global_step)
    
    accelerator.end_training() # finish logging


if __name__ == "__main__":
    main()