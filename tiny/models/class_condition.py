import torch
import torch.nn as nn

from .base import PointCloudDiT


class ClassConditionalPointCloudDiT(PointCloudDiT):
    """
    Point Cloud DiT for Class Conditional Diffusion
    """

    def __init__(
        self,
        input_size: int,
        in_channels: int,
        depth: int,
        hidden_size: int,
        num_heads: int,
        num_classes: int,
        class_embedding_dim: int = None,
        mlp_ratio: int = 4,
        learn_sigma: bool = False,
    ):
        assert class_embedding_dim is not None, "Class embedding dim must be provided"
        super().__init__(
            input_size=input_size,
            in_channels=in_channels,
            depth=depth,
            hidden_size=hidden_size,
            cond_embedding_dim=class_embedding_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            learn_sigma=learn_sigma,
        )

        self.class_embedding = nn.Embedding(
            num_classes, class_embedding_dim
        )  # add 1 for classifier-free guidance

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        classes: torch.Tensor = None,
        **model_kwargs
    ):
        # classes is optional, if not provided, this becomes an unconditional point cloud DiT

        if classes is not None:
            class_embedding = self.class_embedding(classes)
            return super().forward(x, t, cond=class_embedding)
        else:
            return super().forward(x, t)
