from typing import Literal

import torch
from torch.utils.data import Dataset


class SineCosineDataset(Dataset):
    def __init__(self, type: Literal["sine", "cosine", "both"], num_points: int = 64):
        super().__init__()
        self.type = type
        self.num_points = num_points

        self.X = torch.linspace(0, 2 * torch.pi, num_points)
        self.data = []

        if type in {"sine", "both"}:
            self.data.append((torch.sin(self.X), 0))

        if type in {"cosine", "both"}:
            self.data.append((torch.cos(self.X), 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]
