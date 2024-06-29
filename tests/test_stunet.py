import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from medim.models import create_model

class TestSTUNet_small():
    def test_stunet_s_simple_example(self):
        model = create_model("STU-Net-S")
        input_tensor = torch.randn(1, 1, 128, 128, 128)
        output_tensor = model(input_tensor)
        assert output_tensor.shape==torch.Size([1, 105, 128, 128, 128])

    def test_stunet_s_with_local_checkpoint(self):
        model = create_model("STU-Net-S", pretrained=True, 
                            checkpoint_path=os.path.join("tests", "data", "small_ep4k.model"))
        input_tensor = torch.randn(1, 1, 128, 128, 128)
        output_tensor = model(input_tensor)
        assert output_tensor.shape==torch.Size([1, 105, 128, 128, 128])

    def test_stunet_s_without_local_checkpoint(self):
        with pytest.raises(FileNotFoundError):
            create_model("STU-Net-S", pretrained=True, 
                         checkpoint_path=os.path.join("xxx_ep4k.model"))

    def test_stunet_s_with_huggingface_checkpoint(self):
        model = create_model("STU-Net-S", pretrained=True, 
                            checkpoint_path="https://huggingface.co/ziyanhuang/STU-Net/blob/main/small_ep4k.model")
        input_tensor = torch.randn(1, 1, 128, 128, 128)
        output_tensor = model(input_tensor)
        assert output_tensor.shape==torch.Size([1, 105, 128, 128, 128])

class TestSTUNet_base():
    def test_stunet_b_simple_example(self):
        model = create_model("STU-Net-B")
        input_tensor = torch.randn(1, 1, 128, 128, 128)
        output_tensor = model(input_tensor)
        assert output_tensor.shape==torch.Size([1, 105, 128, 128, 128])

    def test_stunet_b_with_huggingface_checkpoint(self):
        model = create_model("STU-Net-B", pretrained=True, 
                            checkpoint_path="https://huggingface.co/ziyanhuang/STU-Net/blob/main/base_ep4k.model")
        input_tensor = torch.randn(1, 1, 128, 128, 128)
        output_tensor = model(input_tensor)
        assert output_tensor.shape==torch.Size([1, 105, 128, 128, 128])

class TestSTUNet_large():
    def test_stunet_b_simple_example(self):
        model = create_model("STU-Net-L")
        input_tensor = torch.randn(1, 1, 128, 128, 128)
        output_tensor = model(input_tensor)
        assert output_tensor.shape==torch.Size([1, 105, 128, 128, 128])

    def test_stunet_b_with_huggingface_checkpoint(self):
        model = create_model("STU-Net-L", pretrained=True, 
                            checkpoint_path="https://huggingface.co/ziyanhuang/STU-Net/blob/main/large_ep4k.model")
        input_tensor = torch.randn(1, 1, 128, 128, 128)
        output_tensor = model(input_tensor)
        assert output_tensor.shape==torch.Size([1, 105, 128, 128, 128])
