from glob import glob
import random
import torch
import math
from torch.utils.data import IterableDataset, get_worker_info
        
class EncodedDataset(IterableDataset):
    """ Generic iterable dataset for *batch encoded* video files stored in folders
    Returns CTHW VAE-encoded videos"""
    def __init__(self, video_folder, remove_last_frame=False, shuffle=True, shuffle_buffer=512, override_length=None):

        self.video_folder = video_folder
        self.remove_last_frame = remove_last_frame
        self.shuffle = shuffle
        self.shuffle_buffer = shuffle_buffer

        self.override_length = override_length

        first_sample = torch.load(glob(self.video_folder)[0], weights_only=True)
        if first_sample.dim() == 4:        
            self.batch_size = 1
        elif first_sample.dim() == 5: 
            self.batch_size = first_sample.shape[0] # assume BS is consistent across all files
        else:
            raise Exception(f"Dataset samples are of shape: {first_sample.shape}. BCTHW or CTHW is required.")
        
        self.samples = self._make_dataset()

    def _make_dataset(self):
        if self.override_length is not None:
            assert self.override_length % self.batch_size == 0, f"Override length {self.override_length} should be multiple of batch size {self.batch_size}"
            samples = glob(self.video_folder)[:(self.override_length // self.batch_size)]
        else:
            samples = glob(self.video_folder)

        if self.shuffle: # shuffles batches
            random.shuffle(samples)
        return samples

    def __len__(self):
        return len(self.samples) * self.batch_size

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = 0
            iter_end = len(self.samples)
        else:
            per_worker = int(math.ceil(len(self.samples) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.samples))

        buffer = []

        for idx in range(iter_start, iter_end):
            video_path = self.samples[idx]
            try:
                video = torch.load(video_path, weights_only=True) # BCTHW
                if video.dim() == 4:
                    video = video.unsqueeze(0) # add batch dimension if not existant
                if self.remove_last_frame:
                    video = video[:, :, :-1, :, :] # BCTHW -> BC(T-1)HW
                
                for b in range(video.shape[0]):
                    if self.shuffle:
                        if len(buffer) < self.shuffle_buffer:
                            buffer.append(video[b])
                        else:
                            # return random samples once buffer is full
                            idx = random.randint(0, self.shuffle_buffer - 1)
                            yield dict(video=buffer[idx]) # CTHW
                            buffer[idx] = video[b]
                    else:
                        yield dict(video=video[b])

            except Exception as e:
                print(f'Error with {e}, {video_path}')
                yield self.__getitem__(random.randint(0, len(self)-1))

        if self.shuffle: # handle remainder
            random.shuffle(buffer)
            for sample in buffer:
                yield dict(video=sample)

    def __getitem__(self, idx):
        video_path = self.samples[idx]
        try:
            video = torch.load(video_path, weights_only=True)
            if video.dim() == 4:
                return dict(video=video) # cthw already
            if self.remove_last_frame:
                video = video[:, :, :-1, :, :]  # BCTHW -> BC(T-1)HW

            # Return a random sample from the batch dimension.
            b = random.randint(0, video.shape[0] - 1)
            return dict(video=video[b]) # CTHW

        except Exception as e:
            print(f'Error with {e}, {video_path}')
            return self.__getitem__(random.randint(0, len(self) - 1))