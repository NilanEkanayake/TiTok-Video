import os
import random
import shutil
import glob
import torch

in_folder = "out_enc"
out_folder = "out_enc_eval"
num_eval = 64

os.makedirs(out_folder, exist_ok=True)

files = glob.glob(os.path.join(in_folder, '*.pt'))
files_to_move = random.sample(files, num_eval//torch.load(files[0]).shape[0]) # assuming batched

for file_path in files_to_move:
    shutil.move(file_path, os.path.join(out_folder, os.path.basename(file_path)))