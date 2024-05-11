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
            self.null_token = nn.Parameter(
                torch.zeros(1, cond_embedding_dim), requires_grad=True
            )

        self.dit_blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio) for i in range(depth)]
        )

        self.out_layer = OutLayer(hidden_size, self.out_channels)

    @property
    def conditional(self):
        return self.cond_embedding_dim is not None

    def cond_embedding(self, batch_size: int, cond=None):
        if self.conditional:
            if cond is not None:
                c = self.c_embed(cond)  # (B, c_dim)
                return c
            else:
                # null token for unconditional generation
                null_token = self.null_token.repeat(batch_size, 1)
                null_token_embedding = self.c_embed(null_token)
                return null_token_embedding
        else:
            return None

    def prepare_cond(self, cond: torch.Tensor):
        """
        Prepare the conditioning vector before calling `forward`.
        """

        return cond

    def _forward_with_cls_and_cond(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cls: torch.Tensor = None,
        cond: torch.Tensor = None,
    ):
        """
        Adds condition embedding to time

        Params:
            - x: data projected to hidden size
            - t: timestep embedding projected to hidden size
            - cond: prepared cond vector, result of cond_embedding
        """
        if cond is not None:
            t = t + cond

        if cls is not None:
            x = torch.cat([cls, x], dim=1)

        # pass thr. dit blocks
        for dit_block in self.dit_blocks:
            x = dit_block(x, t)

        # remove cls tokens
        x = x[:, -self.input_size :, :]

        # final linear layer
        x = self.out_layer(x)

        return x

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor = None,
        **model_kwargs
    ):
        assert (
            x.shape[-2] == self.input_size and x.shape[-1] == self.in_channels
        ), "Input shape mismatch"

        batch_size = x.shape[0]
        x = self.x_embed(x)
        t = self.t_embed(t)
        cond = self.cond_embedding(batch_size, cond)

        # add time and condition embeddings as an extra tokens to input sequence x (from Point E)
        if cond is not None:
            cls_token = torch.cat([t.unsqueeze(1), cond.unsqueeze(1)], dim=1)
        else:
            cls_token = t.unsqueeze(1)

        return self._forward_with_cls_and_cond(x, t, cls=cls_token, cond=cond)
