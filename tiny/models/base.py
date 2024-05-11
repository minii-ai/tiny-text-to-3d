import torch
import torch.nn as nn

from .modules import DiTBlock, OutLayer, TimestepEmbedding


class PointCloudDiT(nn.Module):
    """
    DiT modified for processing point clouds.
    """

    def __init__(
        self,
        input_size: int,
        in_channels: int,
        depth: int,
        hidden_size: int,
        num_heads: int,
        cond_embedding_dim: int = None,
        mlp_ratio: int = 4,
        learn_sigma: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        self.depth = depth
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.cond_embedding_dim = cond_embedding_dim
        self.mlp_ratio = mlp_ratio
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.learn_sigma = learn_sigma

        self.x_embed = nn.Linear(in_channels, hidden_size)
        self.t_embed = TimestepEmbedding(hidden_size)

        if cond_embedding_dim is not None:
            self.c_embed = nn.Linear(cond_embedding_dim, hidden_size)

        self.dit_blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio) for i in range(depth)]
        )

        self.out_layer = OutLayer(hidden_size, self.out_channels)

    @property
    def conditional(self):
        return self.cond_embedding_dim is not None

    def forward_with_cls_and_cond(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cls: torch.Tensor = None,
        cond: torch.Tensor = None,
    ):
        if cls is not None:
            x = torch.cat([cls, x], dim=1)

        # project condition and add it to time embedding
        if cond is not None and self.conditional:
            c = self.c_embed(cond)
            t = t + c

        # pass thr. dit blocks
        for dit_block in self.dit_blocks:
            x = dit_block(x, t)

        # remove cls tokens
        x = x[:, -self.input_size :, :]

        # final linear layer
        x = self.out_layer(x)

        return x

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor = None):
        assert (
            x.shape[-2] == self.input_size and x.shape[-1] == self.in_channels
        ), "Input shape mismatch"

        x = self.x_embed(x)
        t = self.t_embed(t)

        # add time and condition embeddings as an extra tokens to input sequence x (modified technique from Point E)
        cls_token = t.unsqueeze(1)  # (B, 1, h_s)

        return self.forward_with_cls_and_cond(x, t, cls=cls_token, cond=cond)
