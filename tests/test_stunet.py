import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from medim.models.stunet import create_stunet_small

model = create_stunet_small(True, os.path.join(os.path.dirname(__file__), "data\\small_ep4k.model"))
input_tensor = torch.randn(1, 1, 128, 128, 128)
output_tensor = model(input_tensor)
print("Output tensor shape:", output_tensor.shape)