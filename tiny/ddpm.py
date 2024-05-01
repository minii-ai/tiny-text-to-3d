import torch
import torch.nn as nn

from .diffusion import Diffusion
from .dit import PointCloudDiT, UnconditionalPointCloudDiT


class PointCloudDDPM(nn.Module):
    @staticmethod
    def from_config(config: dict):
        model_type = config["model_type"]
        model_config = config["model"]
        diffusion_config = config["diffusion"]

        diffusion = Diffusion(**diffusion_config)
        if model_type == "uncond":
            model = UnconditionalPointCloudDiT(**model_config)

        return PointCloudDDPM(model, diffusion)

    def __init__(self, model: PointCloudDiT, diffusion: Diffusion):

        super().__init__()
        self.model = model
        self.diffusion = diffusion
        self.shape = (model.input_size, model.in_channels)

    def get_loss(self, x_start: torch.Tensor, **model_kwargs):
        return self.diffusion(self.model, x_start, **model_kwargs)

    def forward(
        self,
    ):
        pass