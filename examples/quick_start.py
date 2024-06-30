# -*- encoding: utf-8 -*-
'''
@File    :   quick_start.py
@Time    :   2024/06/10 09:39:29
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   quick start
'''

import torch
import medim


def stunet_simple_example():
    model = medim.create_model("STU-Net-S")
    input_tensor = torch.randn(1, 1, 128, 128, 128)
    output_tensor = model(input_tensor)
    print("Output tensor shape:", output_tensor.shape)


def stunet_with_local_checkpoint():
    model = medim.create_model(
        "STU-Net-S",
        pretrained=True,
        checkpoint_path="../tests/data/Totalseg_small_ep4k.model")
    input_tensor = torch.randn(1, 1, 128, 128, 128)
    output_tensor = model(input_tensor)
    print("Output tensor shape:", output_tensor.shape)

def stunet_with_local_checkpoint_and_args():
    model = medim.create_model(
        "STU-Net-B",
        num_classes=7,
        pretrained=True,
        checkpoint_path="../tests/data/CT_ORG_base_ep1k.model")
    input_tensor = torch.randn(1, 1, 128, 128, 128)
    output_tensor = model(input_tensor)
    print("Output tensor shape:", output_tensor.shape)


def stunet_with_huggingface_checkpoint():
    model = medim.create_model(
        "STU-Net-S",
        pretrained=True,
        checkpoint_path=
        "https://huggingface.co/ziyanhuang/STU-Net/blob/main/small_ep4k.model")
    input_tensor = torch.randn(1, 1, 128, 128, 128)
    output_tensor = model(input_tensor)
    print("Output tensor shape:", output_tensor.shape)


if __name__ == "__main__":
    # stunet_simple_example()
    # stunet_with_local_checkpoint()
    # stunet_with_huggingface_checkpoint()

    stunet_with_local_checkpoint_and_args()
