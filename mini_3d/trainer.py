import torch
from torch.utils.data import DataLoader

from .ddpm import DDPM


class DiffusionTrainer:
    def __init__(
        self,
        ddpm: DDPM,
        train_loader: DataLoader,
        lr: float = 3e-4,
        num_epochs: int = 10,
        resume_checkpoint: bool = False,
        save_dir: str = None,
    ):
        pass
