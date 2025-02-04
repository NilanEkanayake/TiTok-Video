# TiTok for Video Tokenization
### Reconstructions (MSE-loss only, causes blur):
<p>
<img src="assets/recon_1.gif" alt="teaser" width=49%>
<img src="assets/recon_2.gif" alt="teaser" width=49%>
</p>
<p>
<img src="assets/recon_3.gif" alt="teaser" width=49%>
<img src="assets/recon_4.gif" alt="teaser" width=49%>
</p>

### Models:
| Stage | Resolution and FPS | Latent tokens | Codebook Size | Losses | VAE | Model |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 1 | 128x128p, 17 frames | 64 | 15360 | MSE | [checkpoint](https://huggingface.co/chestnutlzj/WF-VAE-L-16Chn) | [checkpoint](https://huggingface.co/NilanE/Titok-Video-Stage1) |
| 2 | 256x256p, 33 frames | 128 | 15360 | MSE | [checkpoint](https://huggingface.co/chestnutlzj/WF-VAE-L-16Chn) | [checkpoint](https://huggingface.co/NilanE/Titok-Video-Stage2) |

### Model eval:
| Model | Eval Res. | SSIM | PSNR | LPIPS |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| WF-VAE 16ch | 128x17 | 0.90 | 26.74 | 0.14 |
| Titok-Video Stage 1 | 128x17 | 0.50 | 19.00 | 0.63 |
| WF-VAE 16ch | 256x33 | 0.91 | 28.09 | 0.16 |
| Titok-Video Stage 2 | 256x33 | 0.54 | 18.59 | 0.62 |

*The models were evaluated on the [MCL_JCV](https://mcl.usc.edu/mcl-jcv-dataset/) dataset. See [eval.ipynb](eval.ipynb) for details.
The TiTok-Video model operates in the latent space of a pre-trained VAE, and therefore the upper bound of its performance is dependant on the VAE used*

### TODO List:
  - [x] First-stage low-res 128x13 training with MSE loss
  - [x] Second-stage 256x33 training with MSE loss
  - [ ] Third-stage 256x33 training with added latent perceptual loss
  - [ ] Fourth-stage 256x33 training with added Adversarial loss (GAN)
  
## Inference:
Install dependencies with:
```
python3 -m pip install -r requirements.txt
```
Then download the checkpoints and VAE for the models from the links above, set the paths accordingly in [the notebook](inference.ipynb) and open in jupyter-lab. From there, run all the cells to reconstruct a demo video.

## Training:
#### Setup:
```
python3 -m pip install -r requirements.txt
```
Then download one of the supported VAEs listed below, and set the parameters accordingly in configs/tiny.yaml and preprocess_dataset/conf.yaml.

#### Dataset pre-processing:
Tweak the config file as desired, then:
```
PYTHONPATH=./ python3 preprocess_dataset/process.py
```
**Notes:**
* A couple WDS-format datasets are linked below, minor code tweaks might be required for some of them.
* A pre-processed dataset made using [vidtok_kl_causal_488_16chn](https://huggingface.co/microsoft/VidTok/blob/main/checkpoints/vidtok_kl_causal_488_16chn.ckpt) can be found [here](https://huggingface.co/datasets/NilanE/vidtok_256_33). *The processed tensors were saved in float32, so a cast might be necessary in the dataset code.*


#### Launch:
```
PYTHONPATH=./ python3 titok_pl.py config=configs/tiny.yaml
```
---

#### Supported VAEs:
```
https://huggingface.co/microsoft/VidTok (can use any of the causal non-quantized models)
https://huggingface.co/chestnutlzj/WF-VAE-L-16Chn
https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.3.0 (8-ch WF-VAE)
https://huggingface.co/THUDM/CogVideoX-2b
```
#### Dataset links:
```
https://huggingface.co/datasets/sailvideo/MiraData-v1
https://huggingface.co/datasets/sailvideo/webvid10m
https://huggingface.co/datasets/Antreas/TALI
https://huggingface.co/datasets/TIGER-Lab/VISTA-400K
https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.1.0
```

#### Uses code from:
```
https://github.com/bytedance/1d-tokenizer
https://github.com/ShivamDuggal4/adaptive-length-tokenizer
https://github.com/lucidrains/vector-quantize-pytorch
https://github.com/PKU-YuanGroup/WF-VAE
https://github.com/microsoft/VidTok
```
