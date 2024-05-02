import torch

from .base import BaseDataset


class SineDataset(BaseDataset):
    def __init__(self, num_points: int = 64, size: int = 128):
        super().__init__()
        self.num_points = num_points
        self.size = size

        X = torch.linspace(-torch.pi, torch.pi, num_points)

        data = torch.stack([X, X.sin()], dim=1)
        self.data = {"data": data, "label": 0, "text": "sine"}

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        return self.data


class CosineDataset(BaseDataset):
    def __init__(self, num_points: int = 64, size: int = 128):
        super().__init__()
        self.num_points = num_points
        self.size = size

        X = torch.linspace(-torch.pi, torch.pi, num_points)

        data = torch.stack([X, X.cos()], dim=1)
        self.data = {"data": data, "label": 1, "text": "cosine"}

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        return self.data


class SineCosineDataset(BaseDataset):
    """Sine and Cosine dataset."""

    def __init__(self, num_points: int, size: int = 128):
        super().__init__()
        self.num_points = num_points
        self.size = size
        self.sine_dataset = SineDataset(num_points, size // 2)
        cosine_size = size - len(self.sine_dataset)
        self.cosine_dataset = CosineDataset(num_points, cosine_size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        if idx < len(self.sine_dataset):
            return self.sine_dataset[idx]
        else:
            return self.cosine_dataset[idx - len(self.sine_dataset)]
