from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from tqdm import tqdm

from ..noise_schedule import NoiseScheduler


class Sampler(ABC, nn.Module):
    """
    Abstract class for sampling steps of a diffusion model.
    """

    def __init__(self, noise_scheduler: NoiseScheduler):
        super().__init__()
        self.noise_scheduler = noise_scheduler
        timesteps = torch.linspace(
            0, noise_scheduler.num_timesteps - 1, noise_scheduler.num_timesteps
        ).int()

        self.register_buffer("timesteps", timesteps)

    @abstractmethod
    def step(self, x_t: torch.Tensor, t: torch.Tensor, cond=None) -> torch.Tensor:
        pass

    @abstractmethod
    def set_timesteps(self, num_timesteps: int):
        pass

    @torch.no_grad()
    def sample_loop(
        self,
        model,
        shape: tuple,
        cond: torch.Tensor = None,
        num_inference_steps: int = None,
        clip_denoised: bool = False,
        guidance_scale: float = 1.0,
        use_cfg: bool = False,
    ):
        for sample in self.sample_loop_progressive(
            model,
            shape,
            cond=cond,
            clip_denoised=clip_denoised,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            use_cfg=use_cfg,
        ):
            pass

        return sample

    def sample_loop_progressive(
        self,
        model,
        shape,
        cond: torch.Tensor = None,
        num_inference_steps: int = None,
        guidance_scale: float = 1.0,
        clip_denoised: bool = False,
        use_cfg: bool = False,
    ):
        """
        Yields sample at each timestep during the sampling loop
        """

        B = shape[0]

        self.set_timesteps(num_inference_steps)

        # start with random gaussian noise
        x_t = torch.randn(*shape, device=self.device)
        timesteps = self.timesteps.flip(0)

        # prepare the condition embedding
        cond = model.prepare_cond(cond)

        # denoise progressively at each timestep
        for t in tqdm(timesteps):
            t = t.repeat(B)
            x_t = self.step(
                model,
                x_t,
                t,
                cond=cond,
                guidance_scale=guidance_scale,
                clip_denoised=clip_denoised,
                use_cfg=use_cfg,
            )
            yield x_t

    def forward(
        self,
        model,
        shape: tuple,
        cond=None,
        clip_denoised: bool = False,
        num_inference_steps: int = 1000,
        guidance_scale: float = 1.0,
        use_cfg: bool = False,
    ) -> torch.Tensor:
        return self.sample_loop(
            model,
            shape,
            cond,
            clip_denoised,
            num_inference_steps,
            guidance_scale,
            use_cfg,
        )
