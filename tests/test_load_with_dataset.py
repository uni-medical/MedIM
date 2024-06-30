import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from medim.models import create_model


class Test_Load_With_Dataset():

    def test_stunet_b_simple_example(self):
        # use the dataset directly to automatically download the ckpt and get PyTorch model
        model = create_model("STU-Net-B", dataset="CT-ORG")
        input_tensor = torch.randn(1, 1, 128, 128, 128)
        output_tensor = model(input_tensor)
        assert output_tensor.shape == torch.Size([1, 8, 128, 128, 128])
