import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce

from .noise_schedule import NoiseScheduler
from .sampler.base import Sampler
from .tensor_utils import extract


class PointCloudDiffusion(nn.Module):
    def __init__(
        self,
        noise_scheduler: NoiseScheduler,
        model,
        sampler: Sampler,
        num_points: int,
        dim: int,
        min_snr_loss_weight: bool = False,
        min_snr_gamma: float = 5.0,
    ):
        super().__init__()
        self.noise_scheduler = noise_scheduler
        self.model = model
        self.sampler = sampler
        self.num_points = num_points
        self.dim = dim

        # https://arxiv.org/pdf/2303.09556
        if min_snr_loss_weight:
            snr = self.noise_scheduler.alphas_cumprod / (
                1 - self.noise_scheduler.alphas_cumprod
            )
            min_snr = torch.clamp(snr, max=min_snr_gamma)
            loss_weights = min_snr / snr
        else:
            loss_weights = torch.ones_like(self.noise_scheduler.betas)

        self.register_buffer("loss_weights", loss_weights)

    def loss(self, x_start: torch.Tensor, cond=None):
        """
        Loss between predicted noise and true noise.
        """

        T = self.noise_scheduler.num_timesteps
        B = x_start.shape[0]

        # add noise to x_start
        t = torch.randint(0, T, (B,)).long().to("cuda")
        noise = torch.randn_like(x_start)
        x_t = self.noise_scheduler.add_noise(x_start, t, noise)

        # predict noise
        if cond is not None:
            pred_noise = self.model(x_t, t, cond)
        else:
            pred_noise = self.model(x_t, t)

        # loss over noise
        mse_losses = F.mse_loss(pred_noise, noise, reduction="none")
        batch_losses = reduce(mse_losses, "b ... -> b", "mean")
        loss_weights = extract(self.loss_weights, t, (B,))
        loss = (loss_weights * batch_losses).mean()

        return loss

    def sample_loop(
        self,
        batch_size: int = 1,
        cond=None,
        clip_denoised: bool = False,
        num_inference_steps: int = 1000,
        guidance_scale: float = 1.0,
        use_cfg: bool = False,
    ):
        shape = (batch_size, self.num_points, self.dim)
        return self.sampler.sample_loop(
            self.model,
            shape=shape,
            cond=cond,
            clip_denoised=clip_denoised,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            use_cfg=use_cfg,
        )

    def sample_loop_progressive(
        self,
        batch_size: int = 1,
        cond=None,
        clip_denoised: bool = False,
        num_inference_steps: int = 1000,
        guidance_scale: float = 1.0,
        use_cfg: bool = False,
    ):

        shape = (batch_size, self.num_points, self.dim)
        return self.sampler.sample_loop_progressive(
            self.model,
            shape=shape,
            cond=cond,
            clip_denoised=clip_denoised,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            use_cfg=use_cfg,
        )

    def forward(
        self,
        batch_size: int = 1,
        cond=None,
        clip_denoised: bool = False,
        num_inference_steps: int = 1000,
        guidance_scale: float = 1.0,
        use_cfg: bool = False,
    ):
        return self.sample_loop(
            batch_size=batch_size,
            cond=cond,
            clip_denoised=clip_denoised,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            use_cfg=use_cfg,
        )
