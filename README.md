# Pytorch-Medical-Image-Models

A collection of PyTorch medical image pre-trained models. 

This repo aims to provide a unified interface for various PyTorch medical image models. 

You can use this cmd to install this toolkit via pip:
```
pip install git+https://github.com/uni-medical/Pytorch-Medical-Image-Models.git
```

Then you can use this repo to get pytorch models like `timm`:
```
import medim

model = medim.create_model("STU-Net-S")

input_tensor = torch.randn(1, 1, 128, 128, 128)
output_tensor = model(input_tensor)
print("Output tensor shape:", output_tensor.shape)
```

The loading of pre-training is still working in progress.
