# TiTok for Video Tokenization

### Models:
TODO

### Model eval:
TODO

## Setup:
Install dependencies with:
```
python3 -m pip install -r requirements.txt
```
  
## Inference:
Run [the eval notebook](inference.ipynb) in jupyter-lab.

## Training:
#### Launch:
```
PYTHONPATH=./ python3 titok_pl.py config=configs/tiny.yaml
```
**Notes:**
* A couple WDS-format datasets are linked below, minor code tweaks might be required for some of them.

---

#### Dataset links:
```
https://huggingface.co/datasets/Vchitect/Vchitect_T2V_DataVerse
https://huggingface.co/datasets/sailvideo/MiraData-v1
https://huggingface.co/datasets/sailvideo/webvid10m
https://huggingface.co/datasets/WenhaoWang/VideoUFO
https://huggingface.co/datasets/TIGER-Lab/VISTA-400K
https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.1.0
```

#### Uses code from:
```
https://github.com/bytedance/1d-tokenizer
https://github.com/ShivamDuggal4/adaptive-length-tokenizer
https://github.com/lucidrains/vector-quantize-pytorch
https://github.com/microsoft/VidTok
```
