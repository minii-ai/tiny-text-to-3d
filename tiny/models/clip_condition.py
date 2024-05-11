import open_clip
import torch

from .base import PointCloudDiT


class CLIPConditionalPointCloudDiT(PointCloudDiT):
    """
    Point Cloud DiT for CLIP Conditional Diffusion
    """

    def __init__(
        self,
        input_size: int,
        in_channels: int,
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
            input_size=input_size,
            in_channels=in_channels,
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

    def forward(self, x: torch.Tensor, t: torch.Tensor, text: list[str] = None):
        if text is not None:
            clip_embedding = self.encode_text(text)
            return super().forward(x, t, cond=clip_embedding)
        else:
            return super().forward(x, t)
