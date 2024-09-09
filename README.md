# MedIM: One-Line Code for Pre-trained Medical Image Models in PyTorch

[![x](https://img.shields.io/badge/Python-3.9|3.10-A7D8FF)]()
[![x](https://img.shields.io/badge/PyTorch-2.4-FCD299)]()

A collection of pre-trained medical image models in PyTorch. This repository aims to provide a unified and easy-to-use interface for comparing and deploying these models.

## Supported Models
- **STU-Net** (`STU-Net-S`, `STU-Net-B`, `STU-Net-L`, `STU-Net-H`) pre-trained on `TotalSegmentator`, `CT-ORG`, `FeTA21`, `BraTS21` (more datasets are WIP).
- **SAM-Med3D** (`SAM-Med3D`) pre-trained on `SA-Med3D-140K`.
- Other pre-trained medical image models are WIP. (You can request support for your model in Issues.)

## Quick Start
### Setup Environment
You can use this cmd to install this toolkit via pip:
```
pip install medim
```
> For developers, you can install in the editable mode via:
> ```
> git clone https://github.com/uni-medical/MedIM.git
> cd MedIM
> pip install -e .
> ```

### Example Usage
First, let us import `medim`.
```
import medim
```
You have four ways to create a PyTorch-compatible model with `create_model`:

**1. use models without pretraining**
```
model = medim.create_model("STU-Net-S") 
```
**2. use local checkpoint**
```
model = medim.create_model(
            "STU-Net-S",
            pretrained=True,
            checkpoint_path="../tests/data/small_ep4k.model") 
```
**3. use checkpoint pre-trained on validated datasets (will automatically download it from HuggingFace)**
```
model = medim.create_model("STU-Net-B", dataset="BraTS21")
```
**4. use HuggingFace url (will automatically download it from HuggingFace)**
```
model = medim.create_model(
            "STU-Net-S",
            pretrained=True,
            checkpoint_path="https://huggingface.co/ziyanhuang/STU-Net/blob/main/small_ep4k.model") 
```
> **Tips**: 
> you can use `MEDIM_CKPT_DIR` environment variable to set custom path for medim model downloading from huggingface.

Then, you can use it as you like.
```
input_tensor = torch.randn(1, 1, 128, 128, 128)
output_tensor = model(input_tensor)
print("Output tensor shape:", output_tensor.shape)
```




More examples are in [examples](https://github.com/uni-medical/MedIM/tree/main/examples).

## Roadmap & TO-DO

- [ ] We will first support more pre-training of STU-Net on different datasets. 
- [ ] The next step is to support more pre-trained medical image models.
- [ ] An easy-to-use interface compatible with MONAI/nnU-Net is still under development. Once developed, you will be able to deploy medical image models more elegantly within the Python/PyTorch ecosystem.

