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
import torch
import time
from utils.utils import AverageMeter, log_grad_norm, save_checkpoint
from utils.metrics import reconstruct_videos, metric_scores, log_codebook_usage

def train_one_epoch(config, logger, accelerator,
                    model, loss_module,
                    optimizer,
                    lr_scheduler,
                    train_dataloader, codebook_dataloader, recon_dataloader,
                    global_step,
                    max_train_steps,
                    pretrained_vae):
    """One epoch training."""

    batch_time_meter = AverageMeter()
    model.train()

    for i, batch in enumerate(train_dataloader):
        start = time.time()

        encoded_videos = batch["video"]

        reconstructed_videos, extra_results_dict = model(encoded_videos)
        autoencoder_loss, loss_dict = loss_module(target=encoded_videos, recon=reconstructed_videos, extra_results_dict=extra_results_dict, global_step=global_step)

        accelerator.backward(autoencoder_loss)

        if config.training.max_grad_norm is not None: # and accelerator.sync_gradients
            accelerator.clip_grad_norm_(loss_module.parameters(), config.training.max_grad_norm)
            
        optimizer.step()
        lr_scheduler.step()

        batch_time_meter.update(time.time() - start)

        # Gather the losses across all processes for logging.
        autoencoder_logs = {}
        for k, v in loss_dict.items():
            autoencoder_logs["train/" + k] = accelerator.gather(v).mean().item()

        # Log gradient norm.
        if ((global_step + 1) % config.experiment.log_grad_norm_every == 0):
            grad_dict = log_grad_norm(model)

            accelerator.log(grad_dict, step=global_step+1)
            
        optimizer.zero_grad(set_to_none=True)

        if (global_step + 1) % config.experiment.log_every == 0:
            samples_per_second_per_gpu = config.training.per_gpu_batch_size / batch_time_meter.val
            lr = lr_scheduler.get_last_lr()[0]
            logger.info(
                f"{samples_per_second_per_gpu:0.2f}/s/gpu "
                f"Batch (t): {batch_time_meter.val:0.4f} "
                f"Step: {global_step + 1} "
                f"Total Loss: {autoencoder_logs['train/total_loss']:0.4f} "
                f"Recon Loss: {autoencoder_logs['train/reconstruction_loss']:0.4f} "
            )
            logs = {"lr/lr": lr, "samples/sec/gpu": samples_per_second_per_gpu, "time/batch_time": batch_time_meter.val}
            logs.update(autoencoder_logs)
            accelerator.log(logs, step=global_step+1)

            # Reset batch / data time meters per log window.
            batch_time_meter.reset()
 
        # Save model checkpoint.
        if (global_step + 1) % config.experiment.save_every == 0:
            save_checkpoint(config, accelerator, global_step+1)
            accelerator.wait_for_everyone()

        # Log videos.
        if (global_step + 1) % config.experiment.generate_every == 0 and accelerator.is_main_process:
            logger.info("Reconstructing videos...")
            reconstruct_videos(model, config, recon_dataloader, pretrained_vae, accelerator, global_step)
            # accelerator.free_memory()
            accelerator.wait_for_everyone()

        # Evaluation.
        if (global_step + 1) % config.experiment.eval_every == 0 and accelerator.is_main_process:
            logger.info(f"Computing metrics...")
            metric_scores(model, config, recon_dataloader, pretrained_vae, accelerator, global_step)

            if not config.model.titok.quant_mode == "vae": # vq mode
                log_codebook_usage(model, config, codebook_dataloader, accelerator, global_step)

            # accelerator.free_memory()
            accelerator.wait_for_everyone()

        global_step += 1

        if global_step >= max_train_steps:
            print(
                f"Finishing training: Global step is >= Max train steps: {global_step} >= {max_train_steps}"
            )
            break

    return global_step
