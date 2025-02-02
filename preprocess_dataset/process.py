import webdataset as wds
import torch
from torch.utils.data import default_collate
from torchvision.transforms import v2, functional
from decord import VideoReader, cpu, bridge
import io
import numpy as np
import uuid
import os
from huggingface_hub import HfFileSystem, hf_hub_url
import time
from omegaconf import OmegaConf
from base_tokenizers import load_vae

def video_process(data, trg_fps, trg_frames, trg_res):
    for sample in data:
        if 'mp4' in sample:
            try:
                bridge.set_bridge('torch')
                transform = v2.Compose([
                    v2.Resize(size=trg_res, interpolation=functional.InterpolationMode.BICUBIC, antialias=True),
                    v2.CenterCrop(size=trg_res)
                ])
                
                if sample['__key__']:
                    out_key = sample['__key__']
                else:
                    out_key = str(uuid.uuid4())
                with io.BytesIO(sample['mp4']) as video_bytes:
                    vr = VideoReader(video_bytes, ctx=cpu(0), num_threads=2)
                    fps = vr.get_avg_fps()
                    vid_dims = vr[0].shape
                    if len(vr) > trg_frames and min(vid_dims[0], vid_dims[1]) >= trg_res and fps >= trg_fps:
                        frame_interval = fps / trg_fps
                        orig_fps_chunk_length = int(trg_frames * frame_interval)
                        num_chunks = len(vr) // orig_fps_chunk_length

                        for i in range(num_chunks):
                            start_idx = i * orig_fps_chunk_length
                            end_idx = start_idx + orig_fps_chunk_length

                            chunk_indices = np.linspace(start_idx, end_idx - 1, trg_frames, dtype=int).tolist()
                            chunk = torch.Tensor(vr.get_batch(chunk_indices))

                            chunk = chunk.permute(0, 3, 1, 2)
                            chunk = transform(chunk)
                            chunk = chunk.permute(1, 0, 2, 3)

                            chunk = chunk.float() / 255
                            chunk = (chunk * 2) - 1

                            yield {'decoded': chunk, '__key__': out_key + f'_{i}'}
            except Exception as error:
                print(f'Decode fail: {error}')

def main():
    config = OmegaConf.load(os.path.join('preprocess_dataset', 'conf.yaml'))

    dtypes = {
        'fp32': torch.float32,
        'bf16': torch.bfloat16,
        'fp16': torch.float16,
    }

    trg_fps = config.dataset.trg_fps
    trg_frames = config.dataset.trg_frames
    trg_res = config.dataset.trg_res
    encode_batch_size = config.vae.batch_size
    dataset_workers = config.dataset.num_workers
    out_dtype = dtypes[config.dataset.out_dtype]
    vae_dtype = dtypes[config.vae.dtype]
    vae_device = config.vae.device
    out_folder = config.dataset.out_folder
    dataset_url = config.dataset.hf_shards
    counter_interval = config.dataset.counter_interval
    out_batch_size = config.dataset.out_batch_size


    vae = load_vae(
        vae_name=config.vae.type,
        embed_dim=config.vae.latent_channels,
        model_path=config.vae.model_path
    ).to(vae_device, vae_dtype)

    fs = HfFileSystem()
    files = [fs.resolve_path(path) for path in fs.glob(dataset_url)]
    shard_list = [hf_hub_url(file.repo_id, file.path_in_repo, repo_type="dataset") for file in files]

    os.makedirs(out_folder, exist_ok=True)

    pipeline = [
            wds.SimpleShardList(shard_list),
            wds.split_by_worker, # no overlapping entries between workers
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            lambda data: video_process(data, trg_fps, trg_frames, trg_res),
            wds.batched(encode_batch_size, partial=False, collation_fn=default_collate),
        ]

    dataset = wds.DataPipeline(*pipeline)
    dataloader = wds.WebLoader(dataset, batch_size=None, pin_memory=True, num_workers=dataset_workers, persistent_workers=True)

    start_time = time.time()
    timer_counter = 0

    chunk_list = []

    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            encoded = vae.encode(batch['decoded'].to(vae_device, vae_dtype))
            for chunk in encoded:
                chunk_list.append(chunk.to('cpu', out_dtype).clone())
                
                if len(chunk_list) == out_batch_size:
                    torch.save(torch.stack(chunk_list).clone(), os.path.join(out_folder, f"{str(uuid.uuid4())}.pt"))
                    chunk_list = []

                timer_counter += 1
                if timer_counter == counter_interval:
                    print(f"IT/S: {(counter_interval)/(time.time()-start_time)}")
                    timer_counter = 0
                    start_time = time.time()

if __name__ == "__main__":
    main()