# TiTok for Video Tokenization

## Inference:
Model:
https://huggingface.co/NilanE/TiTok-Video-64tokens-8Kcodes-exp

Compresses 192p, 16 (13) frame videos to 64 tokens, with a codebook size of ~8000. Since the pretrained VAE used is causal, the three final frames are cut from the reconstruction.

The model was trained for a few hours on a single 4090, with ~80k clips from miradata (see the dataset linked below). The image quality is, frankly, garbage. This is meant as a proof-of-concept more than anything else.

```
python3 -m pip install -r requirements.txt
mkdir -p base_tokenizer/pretrained_model/
wget https://huggingface.co/chestnutlzj/WF-VAE-L-16Chn/resolve/main/diffusion_pytorch_model.safetensors -P base_tokenizer/pretrained_model/
wget https://huggingface.co/chestnutlzj/WF-VAE-L-16Chn/resolve/main/config.json -P base_tokenizer/pretrained_model/
wget https://huggingface.co/NilanE/TiTok-Video-64tokens-8Kcodes-exp/resolve/main/model.pt
```

To use, run inference.ipynb in jupyterlab (or colab, maybe?)

## Training:
Run:
```
python3 -m pip install -r requirements.txt
```
Then download either the CogvideoX VAE, or WF-VAE, and set the path accordingly in configs/tiny.yaml.
The dataset should be video clips pre-encoded with the chosen VAE at the target resolution and frame count, saved to a folder as individual tensors. Make sure to set the max_train_examples in the config to match to dataset size.
I've also uploaded a small dataset here: https://huggingface.co/datasets/NilanE/wfvae-192_17-L16-S10000. However, it has 17 input frames as opposed to 16, so to maintain compatibility, you'll need change:
```
video = torch.load(video_path, weights_only=False)
```
to
```
video = torch.load(video_path, weights_only=False)[:, :-1, :, :] # CTHW -> C(T-1)HW
```
In dataset/encoded.py, or upcomment lines 31, 70, 128 and 185 in model/base/distillation_modules.py as well as change the frame count to 17 in the config. However, the model may/will not train properly with 17 frames, so consider it an experiment.

My code to pre-process the dataset isn't clean enough to publish yet, but it shouldn't be too hard to replicate.
## Uses code from:
```
https://github.com/bytedance/1d-tokenizer
https://github.com/ShivamDuggal4/adaptive-length-tokenizer
https://github.com/lucidrains/vector-quantize-pytorch
https://github.com/PKU-YuanGroup/WF-VAE
https://github.com/zhaoyue-zephyrus/bsq-vit
```
