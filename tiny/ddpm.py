import torch
import torch.nn as nn

from .diffusion import Diffusion
from .dit import PointCloudDiT


class PointCloudDDPM(nn.Module):
    def __init__(self, model: PointCloudDiT, diffusion: Diffusion, shape: tuple):
        assert len(shape) == 2, "Shape must be a tuple of length 2"

        super().__init__()
        self.model = model
        self.diffusion = diffusion
        self.shape = shape

    def get_loss(self, x_start: torch.Tensor, **model_kwargs):
        return self.diffusion(self.model, x_start, **model_kwargs)

    def forward(
        self,
    ):
        pass
