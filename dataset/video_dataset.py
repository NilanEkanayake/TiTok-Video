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
import random

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

def custom_collate(batch): # list of chunks in? Don't collate past that?
    # batch is a list of dicts, want list under a single dict header?
    chunks = [item['video'] for item in batch]
    keys = [item['__key__'] for item in batch]
    fps = [item['fps'] for item in batch]
    return {'video': chunks, 'fps': fps, '__key__': keys}

def video_process(data, fps_range, min_grid, max_grid, patch_size, out_dtype=torch.bfloat16, eval=False):
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
                                # get chunk
                                chunk_num_frames = random.randrange(min_grid[0], max_grid[0]+1, step=patch_size[0])
                                chunk_fps = random.randrange(min_fps, min(max_fps, in_fps) + 1, step=1) # end is not inclusive, so +1?
                                end_idx = start_idx + int(chunk_num_frames * (in_fps / chunk_fps))

                                if in_grid[0] < end_idx: # end condition
                                    break

                                chunk_indices = np.linspace(start_idx, end_idx - 1, chunk_num_frames, dtype=int).tolist()
                                chunk = torch.Tensor(vr.get_batch(chunk_indices))
                                ###

                                # resize chunk

                                max_aspect_ratio = 16//8 # aka 2
                                chunk_height = random.randrange(min_grid[1], min(max_grid[1], in_grid[1])+1, step=patch_size[1])
                                # make sure the ratio check is multiple of patch_size - might be slightly over the max aspect ratio, doesn't matter
                                min_width = max(min_grid[1], chunk_height//max_aspect_ratio - ((chunk_height//max_aspect_ratio) % patch_size[1]))
                                max_width = min(max_grid[1], in_grid[1], int(chunk_height*max_aspect_ratio) - (int(chunk_height//max_aspect_ratio) % patch_size[1]))
                                chunk_width = random.randrange(min_width, max_width+1, step=patch_size[1])
                                
                                if eval: # leave random resolution and frames in to see model's multi-res ability
                                    transform = v2.Compose([
                                        v2.Resize(size=max(chunk_height, chunk_width), interpolation=functional.InterpolationMode.BICUBIC, antialias=True),
                                        v2.CenterCrop(size=(chunk_height, chunk_width))
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


class WebdatasetVideoDataModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        cd = config.dataset
        train_shard_path=cd.train_dataset
        eval_shard_path=cd.eval_dataset

        max_grid = cd.max_grid # THW
        min_grid = cd.min_grid
        fps_range = config.dataset.fps_range

        patch_size = config.model.titok.patch_size # eg. [4, 8, 8]

        train_batch_size = config.training.main.batch_size
        eval_batch_size = config.training.eval.batch_size

        self.eval_samples = config.training.eval.num_eval
        self.num_workers = cd.workers
        self.pin_memory = cd.pin_memory

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

        train_pipeline = [
            wds.ResampledShards(convert_shards(train_shard_path)), # no handler?
            wds.split_by_worker, # no overlapping entries between workers
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(8, handler=wds.warn_and_continue),
            lambda data: video_process(data, fps_range, min_grid, max_grid, patch_size, out_dtype, eval=False),
            wds.shuffle(64, handler=wds.warn_and_continue),
            wds.batched(train_batch_size, partial=False, collation_fn=custom_collate),
        ]

        eval_pipeline = [
            wds.SimpleShardList(convert_shards(eval_shard_path)),
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            lambda data: video_process(data, fps_range, min_grid, max_grid, patch_size, out_dtype, eval=True),
            wds.batched(eval_batch_size, partial=False, collation_fn=custom_collate),
        ]
        
        self.train_dataset = wds.DataPipeline(*train_pipeline)
        self.eval_dataset = wds.DataPipeline(*eval_pipeline)

    
    def train_dataloader(self):
        return wds.WebLoader(self.train_dataset, batch_size=None, pin_memory=self.pin_memory, num_workers=self.num_workers, persistent_workers=True)
    
    def eval_dataloader(self):
        return wds.WebLoader(self.eval_dataset, batch_size=None, pin_memory=self.pin_memory, num_workers=1, persistent_workers=True)