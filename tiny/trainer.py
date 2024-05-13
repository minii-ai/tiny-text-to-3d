import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .diffusion import PointCloudDiffusion
from .lr_scheduler import CosineAnnealingWithWarmupLR
from .utils import count_parameters

device = "cuda" if torch.cuda.is_available() else "cpu"


class PointCloudDiffusionTrainer:
    """
    Trainer for unconditional, class conditional, text conditional, and super resolution point cloud diffusion models.
    """

    def __init__(
        self,
        diffusion: PointCloudDiffusion,
        train_loader: DataLoader,
        init_lr: float = 3e-4,
        lr: float = 3e-4,
        warmup_steps: int = 100,
        num_epochs: int = 10,
        resume_checkpoint: bool = False,
        save_dir: str = None,
        device=device,
        get_batch_fn=None,
        checkpoint_fn=None,
        checkpoint_every: int = 1,
        checkpoint_train_end=None,
    ):
        assert get_batch_fn is not None, "get_batch_fn must be provided"

        self.diffusion = diffusion
        self.train_loader = train_loader
        self.init_lr = init_lr
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.num_epochs = num_epochs
        self.resume_checkpoint = resume_checkpoint
        self.save_dir = save_dir
        self.device = device
        self.get_batch_fn = get_batch_fn
        self.checkpoint_fn = checkpoint_fn
        self.checkpoint_every = checkpoint_every
        self.checkpoint_train_end = checkpoint_train_end

        self.optimizer = torch.optim.Adam(diffusion.parameters(), lr=lr)
        self.scheduler = CosineAnnealingWithWarmupLR(
            init_lr=init_lr,
            optimizer=self.optimizer,
            warmup_steps=warmup_steps,
            T_max=len(train_loader) * num_epochs,
            eta_min=1e-6,
        )

        logs_dir = os.path.join(save_dir, "logs") if save_dir is not None else None
        self.writer = SummaryWriter(logs_dir)

    def train_step(self, batch: dict):
        # get data and cond from batch
        batch = self.get_batch_fn(batch)
        data = batch["data"].to(self.device)
        cond = batch.get("cond", None)

        if hasattr(cond, "to"):
            cond = cond.to(self.device)

        # randomly drop cond for classifier free guidance
        if cond is not None and torch.rand(1).item() < 0.1:
            cond = None

        # get loss
        loss = self.diffusion.loss(data, cond)

        # backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def train(self):
        print(f"[INFO] Model Parameters: {count_parameters(self.diffusion.model)}")

        self.diffusion.train()
        self.diffusion.to(self.device)

        num_iters = len(self.train_loader) * self.num_epochs
        losses = []

        # make save dir
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

        global_step = 0
        with tqdm(total=num_iters) as pbar:
            for epoch in range(self.num_epochs):
                for batch in self.train_loader:
                    loss = self.train_step(batch)
                    pbar.set_postfix(loss=loss.item(), lr=self.scheduler.get_lr()[0])
                    pbar.update(1)
                    losses.append(loss.item())
                    self.scheduler.step()

                    self.writer.add_scalar("train/loss", loss.item(), global_step)
                    self.writer.add_scalar(
                        "lr", self.scheduler.get_lr()[0], global_step
                    )
                    global_step += 1

                if (
                    self.checkpoint_fn is not None
                    and (epoch + 1) % self.checkpoint_every == 0
                ):
                    data = {
                        "epoch": epoch,
                        "diffusion": self.diffusion,
                        "save_dir": self.save_dir,
                        "writer": self.writer,
                        "global_step": global_step,
                    }
                    self.checkpoint_fn(data)

                self.writer.flush()

        if self.save_dir:
            weights_save_path = os.path.join(self.save_dir, "weights.pt")
            torch.save(self.diffusion.state_dict(), weights_save_path)

        if self.checkpoint_train_end:
            self.checkpoint_train_end(
                {
                    "diffusion": self.diffusion,
                    "global_step": global_step,
                    "writer": self.writer,
                }
            )

        return losses
