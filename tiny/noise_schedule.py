from typing import Literal

import torch
import torch.nn as nn

from .tensor_utils import extract

ScheduleType = Literal["linear", "cosine"]


def linear_beta_schedule(num_timesteps: int):
    scale = 1000 / num_timesteps
    beta_start = scale * 1e-4
    beta_end = scale * 0.02

    return torch.linspace(beta_start, beta_end, num_timesteps)


def cosine_beta_schedule(num_timesteps: int):
    s = 0.008
    t = torch.linspace(0, num_timesteps, num_timesteps + 1)
    f = torch.cos((t / num_timesteps + s) / (1 + s) * (torch.pi / 2)) ** 2
    alpha_cumprod = f / f[0]
    beta = 1 - (alpha_cumprod[1:] / alpha_cumprod[:-1])
    beta = torch.clamp(beta, 0, 0.999)

    return beta


def create_noise_schedule(schedule_type: ScheduleType, num_timesteps: int):
    if schedule_type == "linear":
        return linear_beta_schedule(num_timesteps)
    elif schedule_type == "cosine":
        return cosine_beta_schedule(num_timesteps)
    else:
        raise NotImplementedError(f"Schedule type {schedule_type} not implemented.")


class NoiseScheduler(nn.Module):
    """
    Noise Schedule for Diffusion Models. Implements linear and cosine noise schedule from
    Denoising Diffusion Probabilistic Models and Improved Denoising Diffusion Probabilistic Models.
    """

    def __init__(self, num_timesteps: int, schedule_type: ScheduleType = "cosine"):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type

        betas = create_noise_schedule(schedule_type, num_timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod
        )

    def add_noise(
        self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None
    ):
        """
        Adds noise to x_start for t timesteps. Samples x_t ~ q(x_t | x_start).
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        assert noise.shape == x_start.shape

        x_t = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return x_t
