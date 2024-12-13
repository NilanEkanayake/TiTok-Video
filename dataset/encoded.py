from glob import glob
import torch.utils.data as data
import random
import torch

class EncodedDataset(data.Dataset):
    """ Generic dataset for *encoded* videos files stored in folders
    Returns CTHW VAE-encoded videos"""
    def __init__(self, video_folder, train=True):

        self.train = train
        self.video_folder = video_folder

        print('Building datasets...')
        self.samples = self._make_dataset()

    def _make_dataset(self):
        samples = glob(self.video_folder)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path = self.samples[idx]
        try:
            video = torch.load(video_path, weights_only=False)
            return dict(video=video, label="")
        except Exception as e:
            print(f'Error with {e}, {video_path}')
            return self.__getitem__(random.randint(0, self.__len__()-1))