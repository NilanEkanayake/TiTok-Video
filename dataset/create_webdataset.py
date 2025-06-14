from datasets import load_dataset
import tempfile

import ffmpeg
from webdataset import ShardWriter

import os
from huggingface_hub import HfApi

###
out_res = 256 # square
min_fps = 8
min_duration = 2 # seconds

out_crf = 23
out_preset = 'superfast'

videos_per_shard = 5000
repo_id = "NilanE/Vchitect_T2V_DataVerse_256p_8fps_wds"
shard_path = 'shards/%05d.tar'

source_ds = load_dataset("Vchitect/Vchitect_T2V_DataVerse", split="train", streaming=True)
api = HfApi(token="HF_TOKEN")
###

api.create_repo(
    repo_id=repo_id,
    repo_type="dataset",
    exist_ok=True,
    private=False
)

def upload_shard(fname):
    api.upload_file(
        path_or_fileobj=fname,
        path_in_repo=fname,
        repo_type="dataset",
        repo_id=repo_id,
    )
    os.unlink(fname)

with ShardWriter(shard_path, maxcount=videos_per_shard, post=upload_shard) as shard_writer:
    for source_entry in source_ds:
        source_key = source_entry['__key__']
        source_video = source_entry['mp4'] # bytes

        try:
            with tempfile.NamedTemporaryFile(suffix='.mp4') as source_file, tempfile.NamedTemporaryFile(suffix='.mp4') as target_file:
                source_file.write(source_video)

                probe = ffmpeg.probe(source_file.name)

                video = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
                width = int(video['width'])
                height = int(video['height'])
                duration = float(video['duration'])
                fps = float(video['r_frame_rate'].split('/')[0])/float(video['r_frame_rate'].split('/')[1])

                if duration > min_duration and width >= out_res and height >= out_res and fps >= min_fps:
                    if height <= width:
                        crop_type = 'ih'
                        x_offset = (width - height) // 2
                        y_offset = 0
                    else:
                        crop_type = 'iw'
                        x_offset = 0
                        y_offset = (height - width) // 2

                    _ = (
                        ffmpeg.input(source_file.name)
                        .crop(x=x_offset, y=y_offset, width=crop_type, height=crop_type)
                        .filter('scale', width=out_res, height=out_res)
                        .output(target_file.name, format='mp4', codec="libx264", crf=out_crf, preset=out_preset, tune='fastdecode', v='error', an=None, map_metadata=-1)
                        .overwrite_output()
                        .run()
                    )
                    
                    shard_writer.write({
                        '__key__': source_key,
                        'mp4': target_file.read()
                    })

        except Exception as e:
            print(f"Error in video processing:\n{e}")

