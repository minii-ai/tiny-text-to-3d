import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce

from .noise_schedule import NoiseScheduler
from .sampler.base import Sampler


class PointCloudDiffusion(nn.Module):
    def __init__(
        self,
        noise_scheduler: NoiseScheduler,
        model,
        sampler: Sampler,
        num_points: int,
        dim: int,
    ):
        super().__init__()
        self.noise_scheduler = noise_scheduler
        self.model = model
        self.sampler = sampler
        self.num_points = num_points
        self.dim = dim

    def loss(self, x_start: torch.Tensor, cond=None):
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
        loss = reduce(mse_losses, "b ... -> b", "mean").mean()

        return loss

    @torch.inference_mode()
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

    def sample_loop_progressive():
        pass

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
