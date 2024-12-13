import torch
import skimage

from scipy.stats import entropy
import numpy as np
import math
import wandb
import gc


def log_codebook_usage(model, config, dataloader, accelerator, global_step):
    model.eval()

    if config.model.titok.quant_mode == 'fsq':
        codebook_size = math.prod(config.model.titok.fsq_levels)
    else:
        codebook_size = 2 ** config.model.titok.token_size

    codes_frequencies = torch.zeros(codebook_size)
    codes_counter = 0

    with torch.no_grad(), accelerator.autocast():
        for batch in dataloader:
            encoded_videos = batch["video"].to(accelerator.device)
            _, results_dict = accelerator.unwrap_model(model).encode(encoded_videos)

            for tokens in results_dict['codes']:
                codes_frequencies += torch.bincount(tokens, minlength=codebook_size).cpu()
                codes_counter += 1

                if codes_counter == codebook_size:
                    break
            if codes_counter == codebook_size:
                break

    frequencies_np = codes_frequencies.float().numpy()
    codebook_dict = {
        'codebook/usage_percent': (codes_frequencies.count_nonzero() / codebook_size) * 100,
        'codebook/entropy': entropy(frequencies_np / frequencies_np.sum())
    }
    accelerator.log(codebook_dict, step=global_step+1)

    model.train()


def metric_scores(model, config, dataloader, pretrained_vae, accelerator, global_step):
    model.eval()
    vae_gpu = pretrained_vae.to(accelerator.device)

    count = 0
    psnr_scores = []
    ssim_scores = []

    with torch.no_grad(), accelerator.autocast():
        for batch in dataloader:
            encoded_videos = batch["video"].to(accelerator.device)
            recon_videos, _ = accelerator.unwrap_model(model)(encoded_videos)

            original_videos = vae_gpu.decode(encoded_videos)
            decoded_videos = vae_gpu.decode(recon_videos)

            for orig_vid, dec_vid in zip(original_videos, decoded_videos):
                orig_vid_np = orig_vid.permute(1, 2, 3, 0).cpu().float().numpy() # CTHW -> THWC
                dec_vid_np = dec_vid.permute(1, 2, 3, 0).cpu().float().numpy()

                # [0, 1] scaling and clamping done in VAE wrappers
                orig_vid_np = (orig_vid_np * 255).astype(np.uint8)
                dec_vid_np = (dec_vid_np * 255).astype(np.uint8)

                for orig_frame, dec_frame in zip(orig_vid_np, dec_vid_np):
                    psnr_scores.append(skimage.metrics.peak_signal_noise_ratio(orig_frame, dec_frame, data_range=256))
                    ssim_scores.append(skimage.metrics.structural_similarity(orig_frame, dec_frame, channel_axis=-1))

            count += encoded_videos.shape[0]
            if count >= config.training.metric_eval_num:
                break

    model.train()

    result_dict = {'eval/psnr': np.average(np.array(psnr_scores)), 'eval/ssim': np.average(np.array(ssim_scores))} 
    accelerator.log(result_dict, step=global_step+1)


def reconstruct_videos(model, config, dataloader, pretrained_vae, accelerator, global_step):
    model.eval()
    vae_gpu = pretrained_vae.to(accelerator.device)
    recon_count = 0

    with torch.no_grad(), accelerator.autocast():
        for batch in dataloader:
            encoded_videos = batch["video"].to(accelerator.device)
            recon_videos, _ = accelerator.unwrap_model(model)(encoded_videos)

            original_videos = vae_gpu.decode(encoded_videos)
            decoded_videos = vae_gpu.decode(recon_videos)

            for orig_vid, dec_vid in zip(original_videos, decoded_videos):
                dec_vid_reshaped = dec_vid.permute(1, 0, 2, 3)
                orig_vid_reshaped = orig_vid.permute(1, 0, 2, 3)

                merged_video = torch.cat((orig_vid_reshaped, dec_vid_reshaped), dim=-1).cpu().float().numpy() # tch(W) concat
                merged_video = (merged_video * 255).astype(np.uint8)

                recon_count += 1
                accelerator.get_tracker("wandb").log({f"Video recon {recon_count}": wandb.Video(merged_video, fps=config.video_dataset.params.frames_per_second)}, step=global_step+1)

                if recon_count >= config.training.num_generated_videos:
                    break
            if recon_count >= config.training.num_generated_videos:
                break

    model.train()