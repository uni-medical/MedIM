import torch
import medim


if __name__ == "__main__":
    model = medim.create_model("STU-Net-S")
    input_tensor = torch.randn(1, 1, 128, 128, 128)
    output_tensor = model(input_tensor)
    print("Output tensor shape:", output_tensor.shape)
