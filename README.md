# TiTok for Video Tokenization

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
A couple WDS-format datasets are linked below, minor code tweaks might be required for some of them:

#### Training:
```
PYTHONPATH=./ python3 titok_pl.py config=configs/tiny.yaml
```
---

#### Supported VAEs:
```
https://huggingface.co/microsoft/VidTok (can use any of the non-quantized models)
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
