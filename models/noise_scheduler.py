from diffusers import DDPMScheduler


def get_noise_scheduler():
    noise_scheduler = DDPMScheduler.from_pretrained(
        "yisol/IDM-VTON", subfolder="scheduler"
    )
    return noise_scheduler
