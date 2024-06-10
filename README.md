# Pytorch-Medical-Image-Models

A collection of PyTorch medical image pre-trained models. 
This repo aims to provide a unified interface for comparing and deploying these models. 

## Quick Start

You can use this cmd to install this toolkit via pip:
```
pip install git+https://github.com/uni-medical/Pytorch-Medical-Image-Models.git
```
For developer who wanna adding custom models, you can install vis:
```
git clone https://github.com/uni-medical/Pytorch-Medical-Image-Models.git
cd Pytorch-Medical-Image-Models
pip install -e .
```
Then you can use this repo to get pytorch models like `timm`:
```
import medim

model = medim.create_model("STU-Net-S")

input_tensor = torch.randn(1, 1, 128, 128, 128)
output_tensor = model(input_tensor)
print("Output tensor shape:", output_tensor.shape)
```

More examples are in [quick_start](examples\quick_start.py).

## Roadmap

The loading of pre-training is still working in progress.

An ideal version of this repo is like this:
```
import medim

# all-in-one interface to get pretrained pytorch models
model = medim.create_model("STU-Net-S", pretrained_dataset="Totalsegmentator")
# model = medim.create_model("STU-Net-S", pretrained=True, checkpoint_path=<local_path/huggingface_path>)

# torch.transforms for data preprocess
transforms = xxxx

# easy to load data and infer with a medical-image-prefered style
for image in medim.load_image(<image_dir>, transforms=transforms):
    pred = medim.sliding_window_inference(model, image)
    # pred = medim.resize_and_inference(model, image)
    
# then you can validate with your own workflow and metrics
```

