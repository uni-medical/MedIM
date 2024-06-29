# Pytorch-Medical-Image-Models

A collection of PyTorch medical image pre-trained models. 
This repo aims to provide a unified interface for comparing and deploying these models. 

## Quick Start

You can use this cmd to install this toolkit via pip:
```
pip install git+https://github.com/uni-medical/Pytorch-Medical-Image-Models.git
```
> For developer who wanna adding custom models, you can install via:
> ```
> git clone https://github.com/uni-medical/Pytorch-Medical-Image-Models.git
> cd Pytorch-Medical-Image-Models
> pip install -e .
> ```
Then you can use this repo to get pytorch models like `timm`:
```
import medim

# use default setting, without pretrain
model = medim.create_model("STU-Net-S") 

# use local checkpoint
model = medim.create_model("STU-Net-S", pretrained=True, checkpoint_path="../tests/data/small_ep4k.model") 

# use huggingface checkpoint, will download from huggingface
model = medim.create_model("STU-Net-S", pretrained=True, checkpoint_path="https://huggingface.co/ziyanhuang/STU-Net/blob/main/small_ep4k.model") 

input_tensor = torch.randn(1, 1, 128, 128, 128)
output_tensor = model(input_tensor)
print("Output tensor shape:", output_tensor.shape)
```

> if network issues encounted, we recommand use huggingface mirror
> ```
> set HF_ENDPOINT=https://hf-mirror.com (cmd)
> $env:HF_ENDPOINT="https://hf-mirror.com" (powershell)
> ```


More examples are in [quick_start](https://github.com/uni-medical/Pytorch-Medical-Image-Models/blob/main/examples/quick_start.py).

## Roadmap & TO-DO

We will first support more pre-training of STU-Net on different datasets. And the next step is to support more pre-trained medical image models.

A easy-to-use interface compatible for MONAI/nnU-Net is still under development. If developed, you can use a more elegant way to deploy medical image models with Python/PyTorch ecosystem.

