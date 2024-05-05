from models.pipe import get_pipe
from services.tensor import TensorService
import torch
from typing import List


class InferenceService:
    tensor_transform = None
    pipe = None
    garm_img = None
    pose_img = None
    mask = None
    human_img = None
    crop_size = None

    def __init__(self, garm_img, pose_img, mask, human_img, crop_size) -> None:
        self.tensor_transform = TensorService.get_tensor_transform()
        self.pipe = get_pipe()
        self.garm_img = garm_img.convert("RGB").resize((768, 1024))
        self.pose_img = pose_img
        self.mask = mask
        self.human_img = human_img
        self.crop_size = crop_size

    def inferenece(self):
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    images = self.inference_process()
        return images[0].resize(self.crop_size)

    def inference_process(self):
        prompt = "model is wearing clothes"
        negative_prompt = (
            "monochrome, lowres, bad anatomy, worst quality, low quality"
        )
        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )

            prompt = "a photo of a model wearing clothes"
            negative_prompt = (
                "monochrome, lowres, bad anatomy, worst quality, low quality"
            )
            if not isinstance(prompt, List):
                prompt = [prompt] * 1
            if not isinstance(negative_prompt, List):
                negative_prompt = [negative_prompt] * 1
            with torch.inference_mode():
                (
                    prompt_embeds_c,
                    _,
                    _,
                    _,
                ) = self.pipe.encode_prompt(
                    prompt,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                    negative_prompt=negative_prompt,
                )

            pose_img = (
                self.tensor_transform(self.pose_img)
                .unsqueeze(0)
                .to("cuda", torch.float16)
            )
            garm_tensor = (
                self.tensor_transform(self.garm_img)
                .unsqueeze(0)
                .to("cuda", torch.float16)
            )
            generator = torch.Generator("cuda").manual_seed(42)
            return self.pipe(
                prompt_embeds=prompt_embeds.to("cuda", torch.float16),
                negative_prompt_embeds=negative_prompt_embeds.to(
                    "cuda", torch.float16
                ),
                pooled_prompt_embeds=pooled_prompt_embeds.to(
                    "cuda", torch.float16
                ),
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(
                    "cuda", torch.float16
                ),
                num_inference_steps=15,
                generator=generator,
                strength=1.0,
                pose_img=pose_img.to("cuda", torch.float16),
                text_embeds_cloth=prompt_embeds_c.to("cuda", torch.float16),
                cloth=garm_tensor.to("cuda", torch.float16),
                mask_image=self.mask,
                image=self.human_img,
                height=1024,
                width=768,
                ip_adapter_image=self.garm_img.resize((768, 1024)),
                guidance_scale=2.0,
            )[0]
