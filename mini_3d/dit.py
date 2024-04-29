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
        hidden_features = hidden_features or in_features
        out_features = hidden_features or in_features

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
        self.mlp = MLP(hidden_size, hidden_size, act_layer=approx_gelu)
        self.msa = MultiHeadAttention(hidden_size, num_heads)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        """
        Params:
            - x: (B, L, D)
        """

        h = x
        x = self.layernorm1(x)  # (B, L, D)

        # get shift, scale, and gate
        shift_scale_gate = self.adaLN_modulation(c)
        msa_shift, msa_scale, msa_gate, mlp_shift, mlp_scale, mlp_gate = (
            shift_scale_gate.chunk(6, dim=-1)
        )  # (B, D) x 6

        x = self.msa(x)  # multi-head self attention
        x = modulate(x, msa_shift, msa_scale)  # scale and shift
        x = h + msa_gate.unsqueeze(1) * x  # gated residual connection

        h = x
        x = self.layernorm2(x)
        x = self.mlp(x)  # feed forward
        x = modulate(x, mlp_shift, mlp_scale)  # scale and shift
        x = h + mlp_gate.unsqueeze(1)  # gated residual connection

        return x


class DiT(nn.Module):
    """
    DiT modified to operate on 1D sequences and condition on arbitrary embeddings
    instead of class labels.
    """

    def __init__(
        self,
        input_size: int,
        in_channels: int,
        depth: int,
        hidden_size: int,
        num_heads: int,
        cond_embedding_dim: int,
        learn_sigma: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        self.depth = depth
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.cond_embedding_dim = cond_embedding_dim
        self.learn_sigma = learn_sigma

        self.x_embed = nn.Linear(in_channels, hidden_size)
        self.t_embed = TimestepEmbedding(hidden_size)
        self.c_embed = nn.Linear(cond_embedding_dim, hidden_size)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor):
        assert (
            x.shape[-2] == self.input_size and x.shape[-1] == self.in_channels
        ), "Input shape mismatch"

        x = self.x_embed(x)
        t = self.t_embed(t)
        cond = self.c_embed(cond)

        # add time and condition embed
        cond = t + cond
