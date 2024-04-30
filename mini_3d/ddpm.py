import torch
import torch.nn as nn

from .diffusion import Diffusion
from .dit import PointCloudDiT


class DDPM(nn.Module):
    def __init__(self, model: PointCloudDiT, diffusion: Diffusion):
        super().__init__()
        self.model = model
        self.diffusion = diffusion

    def get_loss(self, x_start: torch.Tensor, **model_kwargs):
        return self.diffusion(self.model, x_start, **model_kwargs)

    def forward(
        self,
    ):
        pass
