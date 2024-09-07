import torch
import medim
import numpy as np


def stunet_with_huggingface_checkpoint():
    model = medim.create_model(
        "SAM-Med3D",
        pretrained=True,
        checkpoint_path="https://huggingface.co/blueyo0/SAM-Med3D/blob/main/sam_med3d_turbo.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device", device)
    model = model.to(device)

    input_tensor = torch.randn(1, 1, 128, 128, 128).to(device)
    image_embeddings = model.image_encoder(input_tensor)
    print("Image embeddings shape:", image_embeddings.shape)

    points_coords, points_labels = torch.zeros(1, 0, 3).to(device), torch.zeros(1, 0).to(device)
    new_points_co = torch.Tensor([[64, 64, 64]])
    new_points_la = torch.Tensor([1]).to(torch.int64)
    new_points_co, new_points_la = new_points_co[None].to(device), new_points_la[None].to(device)
    points_coords = torch.cat([points_coords, new_points_co], dim=1)
    points_labels = torch.cat([points_labels, new_points_la], dim=1)

    sparse_embeddings, dense_embeddings = model.prompt_encoder(
        points=[points_coords, points_labels],
        boxes=None,
        masks=None,
    )

    print("Sparse embeddings shape:", sparse_embeddings.shape)
    print("Dense embeddings shape:", dense_embeddings.shape)

    low_res_masks, _ = model.mask_decoder(
        image_embeddings=image_embeddings.to(device),  # (B, 384, 8, 8, 8)
        image_pe=model.prompt_encoder.get_dense_pe(),  # (1, 384, 8, 8, 8)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 384)
        dense_prompt_embeddings=dense_embeddings,  # (B, 384, 8, 8, 8)
    )

    print("Low-res Masks shape/min/max:", low_res_masks.shape, low_res_masks.min(),
          low_res_masks.max())


if __name__ == "__main__":
    stunet_with_huggingface_checkpoint()
