from typing import Literal

import torch
from torch.utils.data import Dataset


class SineCosineDataset(Dataset):
    def __init__(self, type: Literal["sine", "cosine", "both"], num_points: int = 64):
        super().__init__()
        self.type = type
        self.num_points = num_points

        X = torch.linspace(0, 2 * torch.pi, num_points)
        self.data = []

        if type in {"sine", "both"}:
            data = torch.stack([X, X.sin()], dim=1)
            self.data.append((data, 0))

        if type in {"cosine", "both"}:
            data = torch.stack([X, X.cos()], dim=1)
            self.data.append((data, 0))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]
