# MedIM: Pytorch-Medical-Image-Models

A collection of PyTorch medical image pre-trained models. This repository aims to provide a unified interface for comparing and deploying these models.

## Quick Start
### Setup Environment
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

### Example Usage
First, let us import `medim`.
```
import medim
```
You have four ways to create a PyTorch-compatible model with `create_model`:

**1. use default setting, without pretraining**
```
model = medim.create_model("STU-Net-S") 
```
**2. use checkpoint pretrained on validated datasets**
```
model = medim.create_model("STU-Net-B", dataset="BraTS21")
```
**3. use local checkpoint**
```
model = medim.create_model(
            "STU-Net-S",
            pretrained=True,
            checkpoint_path="../tests/data/small_ep4k.model") 
```
**4. use huggingface checkpoint, will download from huggingface**
```
model = medim.create_model(
            "STU-Net-S",
            pretrained=True,
            checkpoint_path="https://huggingface.co/ziyanhuang/STU-Net/blob/main/small_ep4k.model") 
```
Then, you can use it as you like.
```
input_tensor = torch.randn(1, 1, 128, 128, 128)
output_tensor = model(input_tensor)
print("Output tensor shape:", output_tensor.shape)
```

> **Tips**
> If network issues are encountered, we recommend using the Hugging Face mirror:
> ```
> set HF_ENDPOINT=https://hf-mirror.com (cmd)
> $env:HF_ENDPOINT="https://hf-mirror.com" (powershell)
> ```
> Besides, you can use `MEDIM_CKPT_DIR` environment variable to set custom path for medim model downloading from huggingface.


More examples are in [quick_start](https://github.com/uni-medical/Pytorch-Medical-Image-Models/blob/main/examples/quick_start.py).

## Roadmap & TO-DO

We will first support more pre-training of STU-Net on different datasets. The next step is to support more pre-trained medical image models.

An easy-to-use interface compatible with MONAI/nnU-Net is still under development. Once developed, you will be able to deploy medical image models more elegantly within the Python/PyTorch ecosystem.

