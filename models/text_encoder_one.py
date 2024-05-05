import torch
from transformers import CLIPTextModel


def get_text_encoder_one():
    text_encoder_one = CLIPTextModel.from_pretrained(
        "yisol/IDM-VTON",
        subfolder="text_encoder",
        torch_dtype=torch.float16,
    )
    text_encoder_one.requires_grad_(False)
    return text_encoder_one
