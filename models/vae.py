from diffusers import AutoencoderKL
import torch


def get_vae():
    vae = AutoencoderKL.from_pretrained(
        "yisol/IDM-VTON",
        subfolder="vae",
        torch_dtype=torch.float16,
    )
    vae.requires_grad_(False)
    return vae
