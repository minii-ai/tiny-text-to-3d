from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

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


def sample_gaussian(
    mean: torch.Tensor, variance: torch.Tensor, eps: torch.Tensor = None
):
    if eps is None:
        eps = torch.randn_like(mean)
    std = variance**0.5
    sample = mean + std * eps
    return sample


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


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
        one_minus_alpha_cumprod = 1 - alpha_cumprod
        sqrt_one_minus_alpha_cumprod = torch.sqrt(one_minus_alpha_cumprod)

        # values for q posterior, reverse diffusion
        alpha_cumprod_prev = F.pad(alpha_cumprod[:-1], (1, 0), value=1)
        sqrt_alpha_cumprod_prev = torch.sqrt(alpha_cumprod_prev)
        one_minus_alpha_cumprod_prev = 1 - alpha_cumprod_prev

        q_posterior_mean_coef1 = (
            sqrt_alpha_cumprod_prev * beta / one_minus_alpha_cumprod
        )
        q_posterior_mean_coef2 = (
            torch.sqrt(alpha) * one_minus_alpha_cumprod_prev / one_minus_alpha_cumprod
        )
        q_posterior_variance = (
            one_minus_alpha_cumprod_prev / one_minus_alpha_cumprod
        ) * beta

        # register the values as buffers so we can move them to any device easily
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("sqrt_alpha_cumprod", sqrt_alpha_cumprod)
        self.register_buffer("one_minus_alpha_cumprod", one_minus_alpha_cumprod)
        self.register_buffer(
            "sqrt_one_minus_alpha_cumprod", sqrt_one_minus_alpha_cumprod
        )
        self.register_buffer("q_posterior_mean_coef1", q_posterior_mean_coef1)
        self.register_buffer("q_posterior_mean_coef2", q_posterior_mean_coef2)
        self.register_buffer("q_posterior_variance", q_posterior_variance)

    @property
    def device(self):
        return self.beta.device

    def q_mean_variance(self, x_start: torch.Tensor, t: torch.Tensor):
        """
        Computes the mean and variance of q(x_t | x_start)
        """
        mean = extract(self.sqrt_alpha_cumprod, t, x_start.shape) * x_start
        variance = extract(self.one_minus_alpha_cumprod, t, x_start.shape)

        return mean, variance

    def q_sample(
        self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None
    ):
        """
        Adds noise for t timesteps to x_start to get noised sample x_t.
        Equivalent to x_t ~ q(x_t | x_start).
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        mean, variance = self.q_mean_variance(x_start, t)
        x_t = sample_gaussian(mean, variance, noise)

        return x_t

    def predict_x_start(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        """Predict x_start from noise"""
        x_start = (
            x_t - extract(self.sqrt_one_minus_alpha_cumprod, t, x_t.shape) * noise
        ) / extract(self.sqrt_alpha_cumprod, t, x_t.shape)

        return x_start

    def q_posterior_mean_variance(
        self, x_t: torch.Tensor, x_start: torch.Tensor, t: torch.Tensor
    ):
        """
        Get the mean and variance of q(x_t-1 | x_t, x_start)
        """
        mean = (
            extract(self.q_posterior_mean_coef1, t, x_start.shape) * x_start
            + extract(self.q_posterior_mean_coef2, t, x_t.shape) * x_t
        )

        variance = extract(self.q_posterior_variance, t, x_t.shape)

        return mean, variance

    def p_mean_variance(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True,
        **model_kwargs
    ):
        # predict noise
        pred_noise = model(x_t, t, **model_kwargs)

        # predict x_start
        pred_x_start = self.predict_x_start(x_t, t, pred_noise)

        if clip_denoised:
            pred_x_start = pred_x_start.clip(-1, 1)

        mean, variance = self.q_posterior_mean_variance(x_t, pred_x_start, t)
        return mean, variance

    @torch.inference_mode()
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True,
        **model_kwargs
    ):
        """
        Sample from x_t-1 ~ p(x_t-1 | x_t)
        """
        mean, variance = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, **model_kwargs
        )
        nonzero_mask = (t != 0).unsqueeze(-1).unsqueeze(-1)
        eps = torch.randn_like(variance) * nonzero_mask

        prev_x = sample_gaussian(mean, variance, eps)

        return prev_x

    @torch.inference_mode()
    def p_sample_loop(
        self, model: nn.Module, shape: tuple, clip_denoised: bool = True, **model_kwargs
    ):
        for sample in self.p_sample_loop_progressive(
            model=model, shape=shape, clip_denoised=clip_denoised, **model_kwargs
        ):
            final = sample

        return final

    @torch.inference_mode()
    def p_sample_loop_progressive(
        self, model: nn.Module, shape: tuple, clip_denoised: bool = True, **model_kwargs
    ):
        B = shape[0]
        x_t = torch.randn(*shape, device=self.device)

        for t in tqdm(
            reversed(range(self.num_timesteps)),
            total=self.num_timesteps,
            desc="Sampling",
        ):
            t = torch.full((B,), t, device=self.device)
            x_t = self.p_sample(
                model, x_t, t, clip_denoised=clip_denoised, **model_kwargs
            )

            yield x_t

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
        pred = model(x_t, t, **model_kwargs)
        pred_noise = pred

        # mse over pred noise and noise
        loss = F.mse_loss(pred_noise, noise)
        # loss = mean_flat(F.mse_loss(pred_noise, noise, reduction="none")).mean()

        return loss
