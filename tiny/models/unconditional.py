import torch

from .base import PointCloudDiT


class UnconditionalPointCloudDiT(PointCloudDiT):
    """
    Point Cloud DiT for Unconditional Diffusion
    """

    def __init__(
        self,
        input_size: int,
        in_channels: int,
        depth: int,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: int = 4,
        learn_sigma: bool = False,
    ):
        super().__init__(
            input_size=input_size,
            in_channels=in_channels,
            depth=depth,
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            learn_sigma=learn_sigma,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, **model_kwargs):
        return super().forward(x, t)
