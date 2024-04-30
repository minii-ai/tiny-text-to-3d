from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dit import PointCloudDiT


# noise schedule
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


def extract(arr: torch.Tensor, t: torch.Tensor, shape: torch.Size):
    """Extract value from arr at timestep t and make it broadcastable to shape."""
    B = t.shape[0]
    val = arr[t]
    return val.reshape(B, *(1 for _ in range(len(shape) - 1)))


class Diffusion(nn.Module):
    """
    Utilities for noising and denoising during the gaussian diffusion process
    """

    def __init__(
        self,
        schedule_type: Literal["linear", "cosine"] = "linear",
        num_timesteps: int = 1000,
        learn_sigma: bool = False,
    ):
        super().__init__()
        self.schedule_type = schedule_type
        self.num_timesteps = num_timesteps
        self.learn_sigma = learn_sigma

        # create noise schedule and values we'll use during forward and reverse diffusion
        if schedule_type == "linear":
            beta = linear_beta_schedule(num_timesteps)
        elif schedule_type == "cosine":
            beta = cosine_beta_schedule(num_timesteps)

        alpha = 1 - beta
        alpha_cumprod = torch.cumprod(alpha, dim=-1)
        sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod)

        # register the values as buffers so we can move them to any device easily
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("sqrt_alpha_cumprod", sqrt_alpha_cumprod)
        self.register_buffer(
            "sqrt_one_minus_alpha_cumprod", sqrt_one_minus_alpha_cumprod
        )

    @property
    def device(self):
        return self.beta.device

    def q_sample(
        self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None
    ):
        """
        Adds noise for t timesteps to x_start to get noised sample x_t.
        Equivalent to x_t ~ q(x_t | x_start).
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        x_t = (
            extract(self.alpha_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alpha_cumprod, t, x_start.shape) * noise
        )

        return x_t

    def forward(self, model: PointCloudDiT, x_start: torch.Tensor, **model_kwargs):
        """
        Compute simple loss / hybrid loss for diffusion
        """
        device = self.device
        B = x_start.shape[0]

        # uniformly sample timesteps
        t = torch.randint(0, self.num_timesteps, (B,), device=device)

        # add noise to x_start
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)

        # predict noise (assume learn sigma is false for now)
        pred = model(x_t, t)
        pred_noise = pred

        # mse over pred noise and noise
        loss = F.mse_loss(pred_noise, noise)

        return loss
