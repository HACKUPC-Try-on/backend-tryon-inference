from src.unet_hacked_tryon import UNet2DConditionModel
import torch


def get_unet():
    unet = UNet2DConditionModel.from_pretrained(
        "yisol/IDM-VTON",
        subfolder="unet",
        torch_dtype=torch.float16,
    )
    unet.requires_grad_(False)
    return unet
