import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepEmbedding(nn.Module):
    """
    Sinusoidal Timestep Embedding projected to hidden_size
    """

    def __init__(self, hidden_size: int, time_embedding_dim: int = 256):
        super().__init__()
        self.time_embedding_dim = time_embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: torch.Tensor):
        time_embed = timestep_embedding(t, self.time_embedding_dim)
        proj_time_embed = self.mlp(time_embed)

        return proj_time_embed


class MLP(nn.Module):
    """
    MLP layer used in Vision Transformers
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        act_layer=nn.GELU,
        drop: float = 0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor):
        return self.mlp(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        head_dim = dim // num_heads

        assert head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.to_qkv = nn.Linear(dim, 3 * dim)
        self.proj_out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor):
        """
        Params:
            - x: (B, L, D)
        """

        # project to q k v
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # (B, L, D) * 3

        q, k, v = map(
            lambda x: rearrange(
                x, "b l (h d) -> b h l d", h=self.num_heads, d=self.head_dim
            ),
            qkv,
        )  # (B, H, L, head_dim)

        # self attention
        x = (
            F.softmax(q @ k.transpose(-1, -2) / (self.head_dim**0.5), dim=-1) @ v
        )  # (B, H, L, head_dim)

        # concat heads
        x = rearrange(x, "b h l d -> b l (h d)")

        # linear projection
        x = self.proj_out(x)

        return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """
    DiT Block with adaLN-Zero
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: int = 4.0,
    ):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(hidden_size)
        self.layernorm2 = nn.LayerNorm(hidden_size)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size)
        )

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = MLP(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
        )
        self.msa = MultiHeadAttention(hidden_size, num_heads)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        """
        Params:
            - x: (B, L, D)
        """
        # get shift, scale, and gate
        shift_scale_gate = self.adaLN_modulation(c)
        msa_shift, msa_scale, msa_gate, mlp_shift, mlp_scale, mlp_gate = (
            shift_scale_gate.chunk(6, dim=-1)
        )  # (B, D) x 6

        h = x
        x = self.layernorm1(x)  # (B, L, D)
        x = modulate(x, msa_shift, msa_scale)  # scale and shift
        x = self.msa(x)  # multi-head self attention
        x = h + msa_gate.unsqueeze(1) * x  # gated residual connection

        h = x
        x = self.layernorm2(x)
        x = modulate(x, mlp_shift, mlp_scale)  # scale and shift
        x = self.mlp(x)  # feed forward
        x = h + mlp_gate.unsqueeze(1) * x  # gated residual connection

        return x


class OutLayer(nn.Module):
    """
    Final linear projection layer of DiT
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.layernorm = nn.LayerNorm(in_features)
        self.adaLN_modulate = nn.Sequential(
            nn.SiLU(), nn.Linear(in_features, 2 * in_features)
        )
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor):
        x = self.layernorm(x)

        # modulation
        shift, scale = self.adaLN_modulate(x).chunk(2, dim=-1)
        x = scale * x + shift

        # out linear
        x = self.linear(x)

        return x


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

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor = None):
        assert (
            x.shape[-2] == self.input_size and x.shape[-1] == self.in_channels
        ), "Input shape mismatch"

        x = self.x_embed(x)
        t = self.t_embed(t)

        # project condition and add it to time embedding
        if cond is not None and self.conditional:
            c = self.c_embed(cond)
            t = t + c

        # add time and condition embeddings as an extra tokens to input sequence x (modified technique from Point E)
        cls_token = t.unsqueeze(1)  # (B, 1, h_s)
        # x = torch.cat([cls_token, x], dim=1)
        x = x

        # pass thr. dit blocks
        for dit_block in self.dit_blocks:
            x = dit_block(x, t)

        # remove cls tokens
        # x = x[:, 1:, :]

        # final linear layer
        x = self.out_layer(x)

        return x


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

        self.class_embedding = nn.Embedding(num_classes, class_embedding_dim)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, classes: torch.Tensor, **model_kwargs
    ):
        pass
