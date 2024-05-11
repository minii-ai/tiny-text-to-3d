import torch

from .base import PointCloudDiT


class SuperResPointCloudDiT(PointCloudDiT):
    def __init__(
        self,
        input_size: int,
        in_channels: int,
        depth: int,
        hidden_size: int,
        point_dim: int,
        num_heads: int,
        mlp_ratio: int = 4,
        learn_sigma: bool = False,
    ):
        super().__init__(
            input_size=input_size,
            in_channels=in_channels,
            depth=depth,
            hidden_size=hidden_size,
            cond_embedding_dim=point_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            learn_sigma=learn_sigma,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor):
        """
        Params:
            - x: high resolution point cloud (B, N, 3)
            - t: time steps (B, )
            - cond: low resolution point cloud condition (B, M, 3)
        """
