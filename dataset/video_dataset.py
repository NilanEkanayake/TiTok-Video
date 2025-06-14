import torch
import pytorch_lightning as pl
from torchvision.transforms import v2, functional
from decord import VideoReader, cpu, bridge
import io
import numpy as np
import uuid
import webdataset as wds
from torch.utils.data import default_collate
from einops import rearrange
from torch.utils.data import DataLoader
from huggingface_hub import HfFileSystem, hf_hub_url #, get_token
import glob

def convert_shards(shard_paths):
    out_paths = []
    if type(shard_paths) == str:
        shard_paths = [shard_paths]
    for shard_path in shard_paths:
        if shard_path.startswith('hf://'):
            fs = HfFileSystem()
            files = [fs.resolve_path(path) for path in fs.glob(shard_path)]
            urls = [hf_hub_url(file.repo_id, file.path_in_repo, repo_type="dataset") for file in files]
            # shard_path = f"pipe: curl -s -L -H 'Authorization:Bearer {get_token()}' {'::'.join(shard_path)}" # for gated datasets. Add retry/timeout?
            out_paths += urls
        else:
            out_paths.append(shard_path)

    return out_paths

def video_process(data, trg_fps, trg_frames, trg_res, out_dtype=torch.bfloat16, chunk_type='random_multi', max_chunks=20, eval=False):
    for sample in data:
        # for tars made with webdataset's tarwriter, the video key should be 'mp4' and there should also be a __key__ containing a unique id.
        # for tars that are compressed using something like 7zip, the key is read as the file path, eg '/videos/abc.mp4' and there won't be a __key__?

        # setting a lower max_chunks avoids OOMs on long clips, but caps the number of chunks extracted per video.

        # chunk types:
        # random_single -> start from a random index and take a single chunk per video input
        # random_multi -> start from a random index and take multiple sequential chunks per video input. Index chosen is constrained by the 'buffer' of unused frames in a video.
        for video_key in sample.keys():
            if video_key == 'mp4' or video_key.endswith('.mp4'): # allow paths in mp4 key
                try:
                    bridge.set_bridge('torch')
                    if eval:
                        transform = v2.Compose([
                            v2.Resize(size=trg_res, interpolation=functional.InterpolationMode.BICUBIC, antialias=True),
                            v2.CenterCrop(size=trg_res)
                        ])
                    else:   
                        transform = v2.Compose([
                            v2.RandomResizedCrop(size=(trg_res, trg_res), interpolation=functional.InterpolationMode.BICUBIC, antialias=True),
                            v2.RandomHorizontalFlip(p=0.5),                    
                        ])
                    
                    if sample['__key__']:
                        out_key = sample['__key__']
                    else:
                        out_key = str(uuid.uuid4())

                    with io.BytesIO(sample[video_key]) as video_bytes:
                        vr = VideoReader(video_bytes, ctx=cpu(0), num_threads=0) # auto threading
                        fps = vr.get_avg_fps()
                        vid_dims = vr[0].shape
                        num_input_frames = len(vr)
                        if num_input_frames > trg_frames and min(vid_dims[0], vid_dims[1]) >= trg_res and fps >= trg_fps:
                            frame_interval = fps / trg_fps
                            orig_fps_chunk_length = int(trg_frames * frame_interval)
                            num_chunks = num_input_frames // orig_fps_chunk_length

                            start_offset = 0
                            num_chunks = min(num_chunks, max_chunks)

                            if not eval:
                                max_start_idx = 0
                                if chunk_type == 'random_single':
                                    num_chunks = 1
                                    max_start_idx = num_input_frames - orig_fps_chunk_length
                                elif chunk_type == 'random_multi':
                                    max_start_idx = num_input_frames - (num_chunks * orig_fps_chunk_length)

                                if max_start_idx > 0:
                                    start_offset = np.random.randint(0, max_start_idx)
                            else:
                                # num_chunks = 1 # 1 chunk per video for eval diversity
                                pass
                                
                            for i in range(num_chunks):
                                start_idx = (i * orig_fps_chunk_length) + start_offset
                                end_idx = start_idx + orig_fps_chunk_length

                                chunk_indices = np.linspace(start_idx, end_idx - 1, trg_frames, dtype=int).tolist()
                                chunk = torch.Tensor(vr.get_batch(chunk_indices))

                                chunk = chunk.permute(0, 3, 1, 2)
                                chunk = transform(chunk)
                                chunk = chunk.permute(1, 0, 2, 3)

                                chunk = chunk.to(out_dtype) / 255
                                chunk = (chunk * 2) - 1 # [-1, 1]

                                yield {'video': chunk, '__key__': out_key + f'_{i}'}
                except Exception as error:
                    print(f'Decode fail: {error}')


class WebdatasetVideoDataModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        train_shard_path=config.dataset.train_dataset
        eval_shard_path=config.dataset.eval_dataset

        trg_res=config.dataset.resolution
        trg_fps=config.dataset.frames_per_second
        trg_frames=config.dataset.num_frames

        chunk_type=config.dataset.chunk_type
        max_chunks=config.dataset.max_chunks

        train_batch_size=config.training.main.batch_size
        eval_batch_size=config.training.eval.batch_size

        self.eval_samples=config.training.eval.num_eval
        self.num_workers=config.dataset.workers

        self.pin_memory = config.dataset.pin_memory

        dtypes = {
            '16': torch.float16,
            '32': torch.float32,
            '64': torch.float64,
            'bf16': torch.bfloat16,
        }

        if config.training.main.precision == 'transformer-engine':
            out_dtype = torch.bfloat16 # will work?
        else:
            out_dtype = dtypes[config.training.main.precision.split('-')[0]]

        train_pipeline = [
            wds.ResampledShards(convert_shards(train_shard_path)), # no handler?
            wds.split_by_worker, # no overlapping entries between workers
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(8, handler=wds.warn_and_continue),
            lambda data: video_process(data, trg_fps, trg_frames, trg_res, out_dtype, chunk_type, max_chunks, eval=False),
            wds.shuffle(64, handler=wds.warn_and_continue),
            wds.batched(train_batch_size, partial=False, collation_fn=default_collate),
        ]

        eval_pipeline = [
            wds.SimpleShardList(convert_shards(eval_shard_path)),
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            lambda data: video_process(data, trg_fps, trg_frames, trg_res, out_dtype, chunk_type, max_chunks, eval=True),
            wds.batched(eval_batch_size, partial=False, collation_fn=default_collate),
        ]
        
        self.train_dataset = wds.DataPipeline(*train_pipeline)
        self.eval_dataset = wds.DataPipeline(*eval_pipeline)

    
    def train_dataloader(self):
        return wds.WebLoader(self.train_dataset, batch_size=None, pin_memory=self.pin_memory, num_workers=self.num_workers, persistent_workers=True)
    
    def eval_dataloader(self):
        return wds.WebLoader(self.eval_dataset, batch_size=None, pin_memory=self.pin_memory, num_workers=1, persistent_workers=True)