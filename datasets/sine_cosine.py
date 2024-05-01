from typing import Literal

import torch

from .base import BaseDataset


class SineCosineDataset(BaseDataset):
    def __init__(self, type: Literal["sine", "cosine", "both"], num_points: int = 64):
        super().__init__()
        self.type = type
        self.num_points = num_points

        X = torch.linspace(-torch.pi, torch.pi, num_points)
        self.data = []

        if type in {"sine", "both"}:
            data = torch.stack([X, X.sin()], dim=1)
            self.data.append({"data": data, "label": 0, "text": "sine"})

        if type in {"cosine", "both"}:
            data = torch.stack([X, X.cos()], dim=1)
            self.data.append({"data": data, "label": 1, "text": "cosine"})

    def __len__(self):
        # return len(self.data)
        return 128

    def __getitem__(self, idx: int):
        return self.data[0]
        # return self.data[idx]
