import os
from typing import Literal

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .ddpm import PointCloudDDPM
from .utils import count_parameters

device = "cuda" if torch.cuda.is_available() else "cpu"


class PointCloudDiffusionTrainer:
    """
    Trainer for unconditional, class conditional, text conditional, and super resolution point cloud diffusion models.
    """

    def __init__(
        self,
        ddpm: PointCloudDDPM,
        train_loader: DataLoader,
        model_type: Literal["uncond"] = "uncond",
        lr: float = 3e-4,
        num_epochs: int = 10,
        resume_checkpoint: bool = False,
        save_dir: str = None,
        device=device,
        checkpoint_fn=None,
        checkpoint_every: int = 1,
        checkpoint_train_end=None,
        a=False,
    ):
        self.ddpm = ddpm
        self.train_loader = train_loader
        self.model_type = model_type
        self.lr = lr
        self.num_epochs = num_epochs
        self.resume_checkpoint = resume_checkpoint
        self.save_dir = save_dir
        self.device = device
        self.checkpoint_fn = checkpoint_fn
        self.checkpoint_every = checkpoint_every
        self.checkpoint_train_end = checkpoint_train_end
        self.a = a

        self.optimizer = torch.optim.Adam(ddpm.parameters(), lr=lr)

    def train_step(self, batch: dict):
        if self.model_type == "uncond":
            data = batch["data"].to(self.device)
            if self.a:
                data = data.permute(0, 2, 1)
            loss = self.ddpm.get_loss(data)

        # backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def train(self):
        print(f"[INFO] Model Parameters: {count_parameters(self.ddpm)}")

        self.ddpm.train()
        self.ddpm.to(self.device)

        num_iters = len(self.train_loader) * self.num_epochs
        losses = []

        # make save dir
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

        with tqdm(total=num_iters) as pbar:
            for epoch in range(self.num_epochs):
                for batch in self.train_loader:
                    loss = self.train_step(batch)
                    pbar.set_postfix(loss=loss.item())
                    pbar.update(1)
                    losses.append(loss.item())

                if (
                    self.checkpoint_fn is not None
                    and (epoch + 1) % self.checkpoint_every == 0
                ):
                    data = {
                        "epoch": epoch,
                        "ddpm": self.ddpm,
                        "save_dir": self.save_dir,
                    }
                    self.checkpoint_fn(data)

        if self.save_dir:
            weights_save_path = os.path.join(self.save_dir, "weights.pt")
            torch.save(self.ddpm.model.state_dict(), weights_save_path)

        if self.checkpoint_train_end:
            self.checkpoint_train_end({"ddpm": self.ddpm})

        return losses
