from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SamplerStepOutput:
    prev_x: torch.Tensor
    pred_x_start: torch.Tensor
    pred_noise: torch.Tensor


class Sampler(ABC, nn.Module):
    """
    Abstract class for sampling steps of a diffusion model.
    """

    @abstractmethod
    def step(
        self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor = None
    ) -> SamplerStepOutput:
        pass

    @abstractmethod
    def set_timesteps(self, num_timesteps: int):
        pass

    @abstractmethod
    def sample_loop(self):
        pass

    @abstractmethod
    def sample_loop_progressive(self):
        pass

    @abstractmethod
    def forward(
        self,
        model,
        shape: tuple,
        cond: torch.Tensor = None,
        num_inference_steps: int = 1000,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        pass


class DDPMSampler(nn.Module):
    pass


class DDIMSampler(nn.Module):
    pass
