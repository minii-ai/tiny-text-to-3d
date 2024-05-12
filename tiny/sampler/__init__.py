from .base import Sampler
from .ddim import DDIMSampler
from .ddpm import DDPMSampler


def sampler_from_config(config: dict, noise_scheduler):
    config = {**config}
    sampler_type = config["type"]
    del config["type"]

    if sampler_type == "DDPMSampler":
        return DDPMSampler(noise_scheduler, **config)
    elif sampler_type == "DDIMSampler":
        return DDIMSampler(noise_scheduler, **config)
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")
