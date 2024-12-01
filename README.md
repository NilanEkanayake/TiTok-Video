# TiTok for Video Tokenization
Model:
https://huggingface.co/NilanE/TiTok-Video-64tokens-8Kcodes-exp

Compresses 192p, 16 (13) frame videos to 64 tokens, with a codebook size of ~8000. Very poor quality. Since the pretrained VAE used is causal, the three final frames are cur from the reconstruction.

The model was trained for a few hours on a single 4090, with 60k clips from miradata. The image quality is, frankly, garbage. This is meant as a proof-of-concept more than anything else.

The training code wasn't clean enough to release, but I'll add it once I've sorted it through.

## Usage:
```
python3 -m pip install -r requirements.txt
wget https://huggingface.co/chestnutlzj/WF-VAE-L-16Chn/resolve/main/diffusion_pytorch_model.safetensors -P base_tokenizer/pretrained_model/
wget https://huggingface.co/chestnutlzj/WF-VAE-L-16Chn/resolve/main/config.json -P base_tokenizer/pretrained_model/
wget https://huggingface.co/NilanE/TiTok-Video-64tokens-8Kcodes-exp/resolve/main/model.pt
```

To use, run inference.ipynb in jupyterlab (or colab, maybe?)

## Uses code from:
```
https://github.com/bytedance/1d-tokenizer
https://github.com/ShivamDuggal4/adaptive-length-tokenizer
https://github.com/lucidrains/vector-quantize-pytorch
https://github.com/PKU-YuanGroup/WF-VAE
```
