import tempfile
import ffmpeg
from webdataset import ShardWriter
import os
from glob import glob
import uuid

###
# pip install -U webdataset ffmpeg-python 
###

###
out_crf = 23 # https://trac.ffmpeg.org/wiki/Encode/H.264#a1.ChooseaCRFvalue
out_preset = 'superfast' # see https://trac.ffmpeg.org/wiki/Encode/H.264#Preset

videos_per_shard = 512

dataset_path = 'UCF101/train/**/*.avi' # ** means recursive search through all subfolders to find targetted files. Can use .mp4 as well.
shard_path = 'UCF101_WDS/train/%05d.tar'
###

with ShardWriter(shard_path, maxcount=videos_per_shard) as shard_writer:
    for source_path in glob(dataset_path, recursive=True):

        try:
            with tempfile.NamedTemporaryFile(suffix='.mp4') as target_file:
                _ = (
                    ffmpeg.input(source_path)
                    .output(target_file, format='mp4', codec="libx264", crf=out_crf, preset=out_preset, tune='fastdecode', v='error', an=None, map_metadata=-1)
                    .overwrite_output()
                    .run()
                )
                
                shard_writer.write({
                    # '__key__': source_path, # use path as key? Would '/' cause issues? If so, generate random key using uuid.
                    '__key__': str(uuid.uuid4()),
                    'mp4': target_file.read()
                })

        except Exception as e:
            print(f"Error in video processing:\n{e}")