import torch

from .base import PointCloudDiT


class SuperResPointCloudDiT(PointCloudDiT):
    def __init__(
        self,
        input_size: int,
        in_channels: int,
        depth: int,
        low_res_size: int,
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
            cond_embedding_dim=in_channels,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            learn_sigma=learn_sigma,
        )
        self.low_res_size = low_res_size

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, low_res: torch.Tensor, **model_kwargs
    ):
        """
        Params:
            - x: high resolution point cloud (B, N, 3)
            - t: time steps (B, )
            - low_res: low resolution point cloud condition (B, M, 3)
        """

        assert low_res.shape[1] == self.low_res_size, "Low res size mismatch"

        batch_size = x.shape[0]
        x = self.x_embed(x)
        t = self.t_embed(t)
        cond = self.cond_embedding(batch_size, low_res)
        cls_token = torch.cat([t.unsqueeze(1), cond], dim=1)

        return super()._forward_with_cls_and_cond(x, t, cls=cls_token)
