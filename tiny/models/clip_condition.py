import open_clip
import torch

from .base import PointCloudDiT


class CLIPConditionalPointCloudDiT(PointCloudDiT):
    """
    Point Cloud DiT for CLIP Conditional Diffusion
    """

    def __init__(
        self,
        num_points: int,
        dim: int,
        depth: int,
        hidden_size: int,
        num_heads: int,
        clip_model: str = "ViT-B-32",
        mlp_ratio: int = 4,
        learn_sigma: bool = False,
    ):
        clip = open_clip.create_model(clip_model, pretrained="laion2b_s34b_b79k")
        tokenizer = open_clip.get_tokenizer(clip_model)
        clip_embedding_dim = clip.visual.output_dim

        super().__init__(
            num_points=num_points,
            dim=dim,
            depth=depth,
            hidden_size=hidden_size,
            num_heads=num_heads,
            cond_embedding_dim=clip_embedding_dim,
            mlp_ratio=mlp_ratio,
            learn_sigma=learn_sigma,
        )

        self.clip = clip
        self.tokenizer = tokenizer

    @property
    def device(self):
        device = next(self.parameters()).device
        return device

    def encode_text(self, text: list[str]):
        self.clip.eval()
        text = self.tokenizer(text).to(self.device)

        with torch.no_grad():
            text_features = self.clip.encode_text(text)

        return text_features

    def prepare_cond(self, cond: list[str] = None):
        if cond is not None:
            return self.encode_text(cond)
        else:
            return None

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, clip_embedding: torch.Tensor = None
    ):
        if clip_embedding is not None:
            return super().forward(x, t, cond=clip_embedding)
        else:
            return super().forward(x, t)
