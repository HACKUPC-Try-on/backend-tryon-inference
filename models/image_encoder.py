import torch
from transformers import CLIPVisionModelWithProjection


def get_image_encoder():
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "yisol/IDM-VTON",
        subfolder="image_encoder",
        torch_dtype=torch.float16,
    )
    image_encoder.requires_grad_(False)
    return image_encoder
