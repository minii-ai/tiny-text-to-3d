import torch
import torch.nn as nn

from .dit import PointCloudDiT
from .noise_schedule import NoiseScheduler
from .sampler import Sampler


class Diffusion(nn.Module):
    def __init__(
        self,
        noise_scheduler: NoiseScheduler,
        model: PointCloudDiT,
        sampler: Sampler,
        shape: tuple,
    ):
        super().__init__()

    def loss(self, x_start: torch.Tensor):
        pass

    def sample_loop():
        pass

    def sample_loop_progressive():
        pass

    def forward(
        self,
        batch_size: int = 1,
        num_inference_steps: int = 1000,
        guidance_scale: float = 1.0,
    ):
        pass
