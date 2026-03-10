import torch
import lightning as L
from torchvision.transforms import v2, functional
from torchvision import tv_tensors
from decord import VideoReader, cpu, bridge
import io
import numpy as np
from einops import rearrange
from torch.utils.data import Dataset, IterableDataset, DataLoader, default_collate, get_worker_info
import glob
import random
import math
import os
import pandas as pd


def custom_collate(batch): # list of dicts to dict of lists
    keys = batch[0].keys()
    return {k: [b[k] for b in batch] for k in keys}


def get_dtype(config):
    dtypes = {
        '16': torch.float16,
        '32': torch.float32,
        '64': torch.float64,
        'bf16': torch.bfloat16,
    }

    if config.training.main.precision == 'transformer-engine':
        return torch.bfloat16
    else:
        return dtypes[config.training.main.precision.split('-')[0]]


def _video_process(video_paths, config=None, eval=False):
    cs = config.training.sampling

    max_grid = cs.max_grid # THW
    min_grid = cs.min_grid
    fps_range = cs.fps_range
    max_aspect_ratio = cs.max_aspect_ratio
    min_scale = cs.min_scale

    patch_size = config.tokenizer.model.patch_size # eg. [4, 8, 8]
    out_dtype = get_dtype(config)

    # verify sane grid settings
    assert all([dim % ps == 0 for dim, ps in zip(min_grid, patch_size)]) \
        and all([dim % ps == 0 for dim, ps in zip(max_grid, patch_size)]), \
        "dimensions in min_grid and max_grid must be evenly divisible by their respective patch size"
    bridge.set_bridge('torch')

    while True:
        try:
            fpath = random.choice(video_paths)
            vr = VideoReader(fpath, ctx=cpu(0), num_threads=0) # auto threading
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
                    
                    ###
                    chunk_height = random.randrange(min_grid[1], min(max_grid[1], in_grid[1])+1, step=patch_size[1])
                    # make sure the ratio check is multiple of patch_size - might be slightly over the max aspect ratio, doesn't matter
                    width_error = int(chunk_height/max_aspect_ratio) % patch_size[2]
                    min_width = max(min_grid[2], int(chunk_height/max_aspect_ratio) - width_error)
                    max_width = min(max_grid[2], in_grid[2], int(chunk_height*max_aspect_ratio))
                    chunk_width = random.randrange(min_width, max_width+1, step=patch_size[2])
                    ###
                    
                    if eval: # leave random resolution and frames in to see model's multi-res ability
                        transform = v2.Compose([
                            v2.Resize(size=max(chunk_height, chunk_width), interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
                            v2.CenterCrop(size=(chunk_height, chunk_width)),
                        ])
                    else:
                        transform = v2.Compose([
                            v2.RandomResizedCrop(
                                size=(chunk_height, chunk_width),
                                scale=(min_scale, 1.0), # see at least min_scale*100 % of frame
                                ratio=(chunk_width/chunk_height, chunk_width/chunk_height), # fixed ratio - reverse?
                                interpolation=v2.InterpolationMode.BICUBIC, antialias=True,
                            ),
                            v2.RandomHorizontalFlip(p=0.5),                    
                        ])

                    ###
                    chunk_indices = np.linspace(start_idx, end_idx - 1, chunk_num_frames, dtype=int).tolist()
                    chunk = torch.as_tensor(vr.get_batch(chunk_indices)) # .float()

                    chunk = chunk.permute(0, 3, 1, 2)
                    chunk = tv_tensors.Video(chunk) # needed?
                    chunk = transform(chunk)
                    chunk = chunk.permute(1, 0, 2, 3)

                    chunk = chunk.to(out_dtype) / 255
                    chunk = (chunk * 2) - 1 # [-1, 1]
                    ###

                    yield {'video': chunk, 'fps': chunk_fps} 

                    start_idx = end_idx + 1 # setup for next chunk

        except Exception as error:
            print(f'Decode fail: {error}')



def _chunk_buffer(data, buffer_size):
    buffer = []

    for sample in data:
        if len(buffer) < buffer_size: # append
            buffer.append(sample)
        else: # or swap
            idx = random.randrange(buffer_size)
            yield buffer[idx]
            buffer[idx] = sample



def _dynamic_batching(data, config, eval=False):
    cs = config.training.sampling
    patch_size = config.tokenizer.model.patch_size 
    token_range = cs.token_range
    max_grid = cs.max_grid # THW
    max_samples = config.training.eval.eval_samples

    if eval:
        max_seq_len = cs.eval_seq_len
    else:
        max_seq_len = cs.train_seq_len

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
            # chunks['token_counts'] = token_counts
            chunks['token_counts'] = torch.tensor(token_counts, dtype=torch.int32)

            yield chunks # dynamic batch size, keeping under a target sequence length

            chunks = []
            token_counts = []
            curr_seq_len = 0

        curr_seq_len += grid_size + token_count
        chunks.append(sample)
        token_counts.append(token_count)



class CSVVideoDataset(IterableDataset):
    def __init__(self, config, eval=False, buffer_size=64): # if eval, use diff seq_len and ds_path
        self.config = config
        self.eval = eval
        self.buffer_size = buffer_size

        ds_path = config.dataset.eval_dataset if eval else config.dataset.train_dataset
        self.video_paths = pd.read_csv(ds_path)['path'].tolist()


    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is not None:
            random.seed(worker_info.id + worker_info.seed)

        chunks = _video_process(self.video_paths, self.config, eval=self.eval)
        if not self.eval:
            chunks = _chunk_buffer(chunks, self.buffer_size)
        packed = _dynamic_batching(chunks, self.config, eval=self.eval)
        return packed
    


class CSVVideoDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        cd = config.dataset
        self.config = config
        self.num_workers = cd.workers
        self.pin_memory = cd.pin_memory

    def setup(self, stage: str):
        self.train_dataset = CSVVideoDataset(self.config)
        self.eval_dataset = CSVVideoDataset(self.config, eval=True)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=None, pin_memory=self.pin_memory, num_workers=self.num_workers, persistent_workers=True)
    
    def val_dataloader(self):
        return DataLoader(self.eval_dataset, batch_size=None, pin_memory=self.pin_memory, num_workers=1, persistent_workers=True)
    
    def teardown(self, stage: str):
        pass