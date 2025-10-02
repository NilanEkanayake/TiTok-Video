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
import glob
import random
import math

from webdataset.filters import pipelinefilter

def custom_collate(batch): # list of chunks in? Don't collate past that?
    # batch is a list of dicts, want list under a single dict header?
    chunks = [item['video'] for item in batch]
    keys = [item['__key__'] for item in batch]
    fps = [item['fps'] for item in batch]
    return {'video': chunks, 'fps': fps, '__key__': keys}

def _video_process(data, config=None, eval=False):
    cs = config.training.sampling
    cd = config.dataset

    max_grid = cs.max_grid # THW
    min_grid = cs.min_grid
    fps_range = cd.fps_range
    max_aspect_ratio = cd.max_aspect_ratio

    patch_size = config.tokenizer.model.patch_size # eg. [4, 8, 8]
    
    dtypes = {
        '16': torch.float16,
        '32': torch.float32,
        '64': torch.float64,
        'bf16': torch.bfloat16,
    }

    if config.training.main.precision == 'transformer-engine':
        out_dtype = torch.bfloat16
    else:
        out_dtype = dtypes[config.training.main.precision.split('-')[0]]

    # verify sane grid settings
    assert all([dim % ps == 0 for dim, ps in zip(min_grid, patch_size)]) and all([dim % ps == 0 for dim, ps in zip(max_grid, patch_size)]), "dimensions in min_grid and max_grid must be evenly divisible by their respective patch size"

    for sample in data:
        for video_key in sample.keys():
            if video_key == 'mp4' or video_key.endswith('.mp4'): # allow paths in mp4 key
                try:
                    bridge.set_bridge('torch')
                    if sample['__key__']:
                        out_key = sample['__key__']
                    else:
                        out_key = str(uuid.uuid4())

                    with io.BytesIO(sample[video_key]) as video_bytes:
                        vr = VideoReader(video_bytes, ctx=cpu(0), num_threads=0) # auto threading
                        in_fps = int(vr.get_avg_fps())
                        in_grid = [len(vr)] + list(vr[0].shape) # THW
                        min_fps, max_fps = fps_range

                        if all([x >= y for x, y in zip(in_grid, min_grid)]) and in_fps >= min_fps:
                            start_idx = 0
                            end_idx = 0

                            while True:
                                chunk_num_frames = random.randrange(min_grid[0], max_grid[0]+1, step=patch_size[0])
                                chunk_fps = random.randrange(min_fps, min(max_fps, in_fps) + 1, step=1) # end is not inclusive, so +1?

                                end_idx = start_idx + int(chunk_num_frames * (in_fps / chunk_fps))

                                if in_grid[0] < end_idx: # end condition
                                    break

                                chunk_indices = np.linspace(start_idx, end_idx - 1, chunk_num_frames, dtype=int).tolist()
                                chunk = torch.Tensor(vr.get_batch(chunk_indices))
                                
                                chunk_height = random.randrange(min_grid[1], min(max_grid[1], in_grid[1])+1, step=patch_size[1])
                                # make sure the ratio check is multiple of patch_size - might be slightly over the max aspect ratio, doesn't matter
                                width_error = (chunk_height//max_aspect_ratio) % patch_size[2]
                                min_width = max(min_grid[2], chunk_height//max_aspect_ratio - width_error)
                                max_width = min(max_grid[2], in_grid[2], int(chunk_height*max_aspect_ratio) - width_error)
                                chunk_width = random.randrange(min_width, max_width+1, step=patch_size[2])
                                
                                if eval: # leave random resolution and frames in to see model's multi-res ability
                                    transform = v2.Compose([
                                        v2.Resize(size=max(chunk_height, chunk_width), interpolation=functional.InterpolationMode.BICUBIC, antialias=True),
                                        v2.CenterCrop(size=(chunk_height, chunk_width)),
                                    ])
                                else:
                                    transform = v2.Compose([
                                        v2.RandomResizedCrop(size=(chunk_height, chunk_width), interpolation=functional.InterpolationMode.BICUBIC, antialias=True),
                                        v2.RandomHorizontalFlip(p=0.5),                    
                                    ])

                                # need separate resolutions for transforms.
                                chunk = chunk.permute(0, 3, 1, 2)
                                chunk = transform(chunk)
                                chunk = chunk.permute(1, 0, 2, 3)

                                chunk = chunk.to(out_dtype) / 255
                                chunk = (chunk * 2) - 1 # [-1, 1]
                                ###

                                yield {'video': chunk, 'fps': chunk_fps, '__key__': out_key + f'_{start_idx}-{end_idx}'} 

                                start_idx = end_idx + 1 # setup for next chunk

                except Exception as error:
                    print(f'Decode fail: {error}')


def _dynamic_batching(data, config, eval=False):
    patch_size = config.tokenizer.model.patch_size 
    token_range = config.training.sampling.num_token_range
    max_grid = config.training.sampling.max_grid # THW
    max_samples = config.training.eval.num_eval

    if eval:
        max_seq_len = config.training.sampling.eval_seq_len
    else:
        max_seq_len = config.training.sampling.train_seq_len

    assert math.prod(x//y for x, y in zip(max_grid, patch_size)) + token_range[1] <= max_seq_len, "max seq_len (max_grid/patch_size + token_range[1]) must be less than trg_seq_len"

    chunks = []
    token_counts = []
    curr_seq_len = 0
    seen_samples = 0

    for sample in data:
        grid_size = math.prod([x//y for x, y in zip(sample['video'].shape[1:], patch_size)]) # c|THW|
        token_count = random.randrange(token_range[0], token_range[1]+1)

        if eval:
            if seen_samples > max_samples:
                break # return?
            else:
                seen_samples += 1

        if (curr_seq_len + grid_size + token_count) > max_seq_len:
            chunks = custom_collate(chunks)
            chunks['token_count'] = token_counts

            yield chunks # dynamic batch size, keeping under a target sequence length

            chunks = []
            token_counts = []
            curr_seq_len = 0

        curr_seq_len += grid_size + token_count
        sample['grid_size'] = grid_size
        chunks.append(sample)
        token_counts.append(token_count)

video_process = pipelinefilter(_video_process)
dynamic_batching = pipelinefilter(_dynamic_batching)

class WebdatasetVideoDataModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        cd = config.dataset
        train_shard_path=cd.train_dataset
        eval_shard_path=cd.eval_dataset

        self.num_workers = cd.workers
        self.pin_memory = cd.pin_memory

        train_pipeline = [
            wds.ResampledShards(train_shard_path), # no handler?
            wds.split_by_worker, # no overlapping entries between workers
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(8, handler=wds.warn_and_continue),
            video_process(config, eval=False),
            wds.shuffle(64, handler=wds.warn_and_continue),
            dynamic_batching(config, eval=False),
        ]

        eval_pipeline = [
            wds.SimpleShardList(eval_shard_path),
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            video_process(config, eval=True),
            dynamic_batching(config, eval=True),
        ]
        
        self.train_dataset = wds.DataPipeline(*train_pipeline)
        self.eval_dataset = wds.DataPipeline(*eval_pipeline)

    
    def train_dataloader(self):
        return wds.WebLoader(self.train_dataset, batch_size=None, pin_memory=self.pin_memory, num_workers=self.num_workers, persistent_workers=True)
    
    def eval_dataloader(self):
        return wds.WebLoader(self.eval_dataset, batch_size=None, pin_memory=self.pin_memory, num_workers=1, persistent_workers=True)