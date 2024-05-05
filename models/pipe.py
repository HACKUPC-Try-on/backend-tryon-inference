from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from models.unet import get_unet
from models.vae import get_vae
from transformers import CLIPImageProcessor
from models.text_encoder_one import get_text_encoder_one
from models.text_encoder_two import get_text_encoder_two
from models.tokenizer_one import get_tokenizer_one
from models.tokenizer_two import get_tokenizer_two
from models.noise_scheduler import get_noise_scheduler
from models.image_encoder import get_image_encoder
from models.unet_encoder import get_unet_encoder
import torch


def get_pipe():
    pipe = TryonPipeline.from_pretrained(
        "yisol/IDM-VTON",
        unet=get_unet(),
        vae=get_vae(),
        feature_extractor=CLIPImageProcessor(),
        text_encoder=get_text_encoder_one(),
        text_encoder_2=get_text_encoder_two(),
        tokenizer=get_tokenizer_one(),
        tokenizer_2=get_tokenizer_two(),
        scheduler=get_noise_scheduler(),
        image_encoder=get_image_encoder(),
        torch_dtype=torch.float16,
    )
    pipe.unet_encoder = get_unet_encoder()
    pipe.to("cuda")
    pipe.unet_encoder.to("cuda")
    return pipe
