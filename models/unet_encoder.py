from src.unet_hacked_garmnet import (
    UNet2DConditionModel as UNet2DConditionModel_ref,
)
import torch


def get_unet_encoder():
    unet_encoder = UNet2DConditionModel_ref.from_pretrained(
        "yisol/IDM-VTON",
        subfolder="unet_encoder",
        torch_dtype=torch.float16,
    )
    unet_encoder.requires_grad_(False)
    return unet_encoder
