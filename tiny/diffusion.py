from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from tqdm import tqdm

ScheduleType = Literal["linear", "cosine"]


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


def create_noise_schedule(schedule_type: ScheduleType, num_timesteps: int):
    if schedule_type == "linear":
        return linear_beta_schedule(num_timesteps)
    elif schedule_type == "cosine":
        return cosine_beta_schedule(num_timesteps)
    else:
        raise NotImplementedError(f"Schedule type {schedule_type} not implemented.")


def extract(arr: torch.Tensor, t: torch.Tensor, shape: torch.Size):
    """Extract value from arr at timestep t and make it broadcastable to shape."""
    B = t.shape[0]
    val = arr[t]
    return val.reshape(B, *(1 for _ in range(len(shape) - 1)))


def make_broadcastable(val: torch.Tensor, shape: torch.Size):
    """Expand val to be broadcastable to shape. Unsqueezes dimensions to the right."""
    while val.dim() < len(shape):
        val = val.unsqueeze(-1)
    return val


class Diffusion(nn.Module):
    def __init__(self, schedule_type: ScheduleType, num_timesteps: int):
        super().__init__()
        self.schedule_type = schedule_type
        self.num_timesteps = num_timesteps

        betas = create_noise_schedule(schedule_type, num_timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

        # values for q posterior
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], pad=(1, 0), value=1)
        q_posterior_mean_coef1 = (torch.sqrt(alphas_cumprod_prev) * betas) / (
            1 - alphas_cumprod
        )
        q_posterior_mean_coef2 = (torch.sqrt(alphas) * (1 - alphas_cumprod_prev)) / (
            1 - alphas_cumprod
        )
        q_posterior_variance = (1 - alphas_cumprod_prev) * betas / (1 - alphas_cumprod)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod
        )
        self.register_buffer("q_posterior_mean_coef1", q_posterior_mean_coef1)
        self.register_buffer("q_posterior_mean_coef2", q_posterior_mean_coef2)
        self.register_buffer("q_posterior_variance", q_posterior_variance)

    @property
    def device(self):
        return self.betas.device

    def q_sample(
        self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None
    ):
        """
        Add noise to x_start for t timesteps. Samples x_t ~ q(x_t | x_start).
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        assert noise.shape == x_start.shape

        x_t = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return x_t

    def q_posterior_mean_variance(
        self, x_t: torch.Tensor, x_start: torch.Tensor, t: torch.Tensor
    ):
        """
        Gets the mean and variance of q(x_{t-1} | x_t, x_start)
        """

        mean = (
            extract(self.q_posterior_mean_coef1, t, x_start.shape) * x_start
            + extract(self.q_posterior_mean_coef2, t, x_t.shape) * x_t
        )
        variance = extract(self.q_posterior_variance, t, x_t.shape)

        return mean, variance

    def predict_x_start(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        """
        Predicts x_start from x_t and noise.
        """

        x_start = (
            x_t - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * noise
        ) / extract(self.sqrt_alphas_cumprod, t, x_t.shape)

        return x_start

    def p_sample(
        self,
        model,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor = None,
        guidance_scale: float = 1.0,
        clip_denoised: bool = False,
    ):
        """
        Samples x_{t-1} ~ p(x_{t-1} | x_t, t) or p(x_{t-1} | x_t, t, c) if cond is provided.
        """
        if cond is not None:
            # predict unconditional and conditional noise
            pred_cond_noise = model(x_t, t, cond)
            pred_uncond_noise = model(x_t, t)

            # classifier-free guidance
            pred_noise = (
                1 - guidance_scale
            ) * pred_uncond_noise + guidance_scale * pred_cond_noise

        else:
            # predict the noise added to x_start
            pred_noise = model(x_t, t)

        # predict x_start
        pred_x_start = self.predict_x_start(x_t, t, pred_noise)

        if clip_denoised:
            pred_x_start = pred_x_start.clip(-1.0, 1.0)

        # get mean and variance from q(x_{t-1} | x_t, pred_x_start)
        mean, variance = self.q_posterior_mean_variance(x_t, pred_x_start, t)

        # zero out variance at timestep t = 0
        nonzero_mask = make_broadcastable(t != 0, variance.shape)

        # sample
        eps = torch.randn_like(x_t)
        prev_x = mean + variance**0.5 * nonzero_mask * eps

        return prev_x

    @torch.no_grad()
    def p_sample_loop(
        self,
        model,
        shape: tuple,
        cond: torch.Tensor = None,
        clip_denoised: bool = False,
    ):
        for sample in self.p_sample_loop_progressive(
            model, shape, cond=cond, clip_denoised=clip_denoised
        ):
            pass

        return sample

    @torch.no_grad()
    def p_sample_loop_progressive(
        self,
        model,
        shape,
        cond: torch.Tensor = None,
        guidance_scale: float = 1.0,
        clip_denoised: bool = False,
    ):
        """
        Yields sample at each timestep during the sampling loop
        """

        B = shape[0]
        x_t = torch.randn(*shape, device=self.device)
        timesteps = range(self.num_timesteps)[::-1]

        for t in tqdm(timesteps):
            t = torch.full((B,), t, device=self.device)
            x_t = self.p_sample(
                model,
                x_t,
                t,
                cond=cond,
                guidance_scale=guidance_scale,
                clip_denoised=clip_denoised,
            )
            yield x_t

    def training_losses(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor = None,
    ):
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)

        if cond is not None:
            pred_noise = model(x_t, t, cond)
        else:
            pred_noise = model(x_t, t)

        mse_losses = F.mse_loss(pred_noise, noise, reduction="none")
        loss = reduce(mse_losses, "b ... -> b", "mean").mean()

        return loss
