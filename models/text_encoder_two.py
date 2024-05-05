import torch
from transformers import CLIPTextModelWithProjection


def get_text_encoder_two():
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        "yisol/IDM-VTON",
        subfolder="text_encoder_2",
        torch_dtype=torch.float16,
    )
    text_encoder_two.requires_grad_(False)
    return text_encoder_two
