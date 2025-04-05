# Model inference setup and guide:

## Installation:
```bash
git clone https://github.com/NilanEkanayake/TiTok-Video
cd TiTok-Video
python3 -m pip install -r requirements.txt
python3 -m pip install jupyterlab
```

## Running the notebook:
From within the TiTok-Video directory, launch jupyter-lab:
```bash
jupyter lab --no-browser
```
Then, copy the printed local URL and open it in a browser. The link should look something like ```http://localhost:8888/lab?token=aabbcc...```
From there, open [inference.ipynb](inference.ipynb) through the file explorer on the left, and once open, change the paths to the config and checkpoint from https://huggingface.co/NilanE/TiTok-Video-128p-32f-128tok or a model you've trained.

There are two main sections in the inference notebook. The first is for a single tokenization and reconstruction from an input video. The second splits the input video into chunks and encodes them separately, before stitching them back together and saving the result. The below should work with either.

## Tokenizing a video:
Change the input and output paths in the tokenize_and_reconstruct function call:
```python
tokenize_and_reconstruct("path_to_input.mp4", "path_to_output.mp4")
```
And then run the cells from the top down until you run the function call. Cells can be run by selecting them and then pressing 'shift+enter'. If using the long video tokenization portion of the notebook, skip running the first tokenize_and_reconstruct.

The input video will be center cropped and downscaled to the model's expected resolution, framerate and number of frames, truncating the video's duration if necessary.
The video will then be tokenized, reconstructed, and lastly width-concatenated with the original input before being saved. The width concatenation is for side-by-side comparison between the original video and the reconstruction.

To only save the reconstruction, change the following in the tokenization function:
```python
# change
merged_video = torch.cat((orig, recon), dim=-1).permute(1, 2, 3, 0).cpu().float().numpy() # cth(W) concat -> thwc
# to
merged_video = recon.permute(1, 2, 3, 0).cpu().float().numpy() # cthw -> thwc
```

