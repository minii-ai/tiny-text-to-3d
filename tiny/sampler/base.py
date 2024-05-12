from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn

from ..noise_schedule import NoiseScheduler


class Sampler(ABC, nn.Module):
    """
    Abstract class for sampling steps of a diffusion model.
    """

    def __init__(self, noise_scheduler: NoiseScheduler):
        super().__init__()
        self.noise_scheduler = noise_scheduler

    # @abstractmethod
    def step(self, x_t: torch.Tensor, t: torch.Tensor, cond=None) -> torch.Tensor:
        pass

    # @abstractmethod
    def set_timesteps(self, num_timesteps: int):
        pass

    # @abstractmethod
    def sample_loop(
        self,
        model,
        shape: tuple,
        cond=None,
        clip_denoised: bool = False,
        num_inference_steps: int = 1000,
        guidance_scale: float = 1.0,
        use_cfg: bool = False,
    ):
        pass

    # @abstractmethod
    def sample_loop_progressive(
        self,
        model,
        shape: tuple,
        cond=None,
        clip_denoised: bool = False,
        num_inference_steps: int = 1000,
        guidance_scale: float = 1.0,
        use_cfg: bool = False,
    ):
        pass

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
