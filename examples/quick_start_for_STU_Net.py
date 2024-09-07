import torch
import medim


def stunet_simple_example():
    model = medim.create_model("STU-Net-S")
    input_tensor = torch.randn(1, 1, 128, 128, 128)
    output_tensor = model(input_tensor)
    print("Output tensor shape:", output_tensor.shape)


def stunet_with_local_checkpoint():
    model = medim.create_model("STU-Net-S",
                               pretrained=True,
                               checkpoint_path="../tests/data/Totalseg_small_ep4k.model")
    input_tensor = torch.randn(1, 1, 128, 128, 128)
    output_tensor = model(input_tensor)
    print("Output tensor shape:", output_tensor.shape)


def stunet_with_huggingface_checkpoint():
    model = medim.create_model(
        "STU-Net-S",
        pretrained=True,
        checkpoint_path="https://huggingface.co/ziyanhuang/STU-Net/blob/main/small_ep4k.model")
    input_tensor = torch.randn(1, 1, 128, 128, 128)
    output_tensor = model(input_tensor)
    print("Output tensor shape:", output_tensor.shape)


def stunet_with_local_checkpoint_and_args():
    model = medim.create_model("STU-Net-B",
                               input_channels=4,
                               num_classes=5,
                               strides=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                               pretrained=True,
                               checkpoint_path="../tests/data/BraTS21_base_ep1k.model")
    input_tensor = torch.randn(1, 4, 128, 128, 128)
    output_tensor = model(input_tensor)
    print("Output tensor shape:", output_tensor.shape)


def stunet_with_dataset_name_Totalseg():
    model = medim.create_model("STU-Net-B", dataset="TotalSegmentator")
    input_tensor = torch.randn(1, 1, 128, 128, 128)
    output_tensor = model(input_tensor)
    print("Output tensor shape:", output_tensor.shape)


def stunet_with_dataset_name_BraTS21():
    model = medim.create_model("STU-Net-B", dataset="BraTS21")
    input_tensor = torch.randn(1, 4, 128, 128, 128)
    output_tensor = model(input_tensor)
    print("Output tensor shape:", output_tensor.shape)


def stunet_with_local_checkpoint_cuda():
    model = medim.create_model("STU-Net-S",
                               pretrained=True,
                               checkpoint_path="../tests/data/Totalseg_small_ep4k.model").cuda()
    input_tensor = torch.randn(1, 1, 128, 128, 128).cuda()
    output_tensor = model(input_tensor)
    print("Output tensor shape:", output_tensor.shape)


if __name__ == "__main__":
    # stunet_simple_example()
    # stunet_with_local_checkpoint()
    # stunet_with_huggingface_checkpoint()

    # stunet_with_local_checkpoint_and_args()
    # stunet_with_dataset_name_Totalseg()
    # stunet_with_dataset_name_BraTS21()

    stunet_with_local_checkpoint_cuda()
