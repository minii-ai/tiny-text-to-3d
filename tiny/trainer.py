import logging
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .diffusion import PointCloudDiffusion
from .lr_scheduler import CosineAnnealingWithWarmupLR
from .utils import count_parameters

device = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


class PointCloudDiffusionTrainer:
    """
    Trainer for unconditional, class conditional, text conditional, and super resolution point cloud diffusion models.
    """

    def __init__(
        self,
        diffusion: PointCloudDiffusion,
        train_loader: DataLoader,
        lr: float = 3e-4,
        train_steps: int = 10000,
        warmup_steps: int = 100,
        resume_checkpoint: bool = False,
        save_dir: str = None,
        device=device,
        get_batch_fn=None,
        checkpoint_fn=None,
        checkpoint_every: int = 1,
        checkpoint_train_end=None,
    ):
        assert get_batch_fn is not None, "get_batch_fn must be provided"

        self.diffusion = diffusion.to(device)
        self.train_loader = train_loader
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.train_steps = train_steps
        self.resume_checkpoint = resume_checkpoint
        self.save_dir = save_dir
        self.device = device
        self.get_batch_fn = get_batch_fn
        self.checkpoint_fn = checkpoint_fn
        self.checkpoint_every = checkpoint_every
        self.checkpoint_train_end = checkpoint_train_end

        self.optimizer = torch.optim.Adam(diffusion.parameters(), lr=lr)
        self.lr_scheduler = CosineAnnealingWithWarmupLR(
            optimizer=self.optimizer,
            warmup_steps=warmup_steps,
            T_max=train_steps,
            eta_min=1e-6,
        )

        logs_dir = os.path.join(save_dir, "logs") if save_dir is not None else None
        self.writer = SummaryWriter(logs_dir)

        self.global_step = 1
        self.start_step = 1

        if self.resume_checkpoint:
            self.sync_checkpoint()

    def sync_checkpoint(self):
        logging.info("Syncing from checkpoint...")
        checkpoint_dirs = [
            d for d in os.listdir(self.save_dir) if d.startswith("checkpoint")
        ]

        checkpoint_iters = sorted(
            [int(checkpoint_dir.split("-")[-1]) for checkpoint_dir in checkpoint_dirs]
        )

        if len(checkpoint_iters) > 0:
            self.start_step = checkpoint_iters[-1] + 1
            checkpoint_path = os.path.join(
                self.save_dir, f"checkpoint-{checkpoint_iters[-1]}", "checkpoint.pt"
            )

            checkpoint_data = torch.load(checkpoint_path, map_location=self.device)
            weights = checkpoint_data["weights"]
            optimizer = checkpoint_data["optimizer"]
            lr_scheduler = checkpoint_data["lr_scheduler"]

            logging.info(
                f"Loading state dict for diffusion, optimizer, and lr_scheduler from checkpoint {checkpoint_iters[-1]}"
            )
            self.diffusion.load_state_dict(weights)
            self.optimizer.load_state_dict(optimizer)
            self.lr_scheduler.load_state_dict(lr_scheduler)

            # raise RuntimeError

    def train_step(self, batch: dict):
        self.diffusion.train()
        # get data and cond from batch -> get_batch_fn should return a dict with keys "data" and "cond"
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
        self.lr_scheduler.step()

        return loss

    def train(self):
        logging.info(f"Model Parameters: {count_parameters(self.diffusion.model)}")
        logging.info(f"Train Steps: {self.train_steps}")
        logging.info(f"Dataset Size: {len(self.train_loader.dataset)}")

        self.diffusion.to(self.device)

        losses = []

        # make save dir
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

        with tqdm(total=self.train_steps) as pbar:
            for batch in self.train_loader:
                # start training at start step
                if self.global_step < self.start_step:
                    self.global_step += 1
                    pbar.update(1)
                    continue

                loss = self.train_step(batch)
                losses.append(loss.item())

                pbar.set_postfix(loss=loss.item(), lr=self.lr_scheduler.get_lr()[0])
                pbar.update(1)

                self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                self.writer.add_scalar(
                    "train/lr", self.lr_scheduler.get_lr()[0], self.global_step
                )

                if self.global_step % self.checkpoint_every == 0:
                    checkpoint_dir = os.path.join(
                        self.save_dir, f"checkpoint-{self.global_step}"
                    )
                    checkpoint_save_path = os.path.join(checkpoint_dir, "checkpoint.pt")

                    # save checkpoint
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    checkpoint_data = {
                        "weights": self.diffusion.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "lr_scheduler": self.lr_scheduler.state_dict(),
                    }
                    torch.save(checkpoint_data, checkpoint_save_path)

                    # call custom checkpoint function
                    if self.checkpoint_fn is not None:
                        data = {
                            "diffusion": self.diffusion,
                            "save_dir": self.save_dir,
                            "writer": self.writer,
                            "step": self.global_step,
                            "optimizer": self.optimizer,
                            "lr_scheduler": self.lr_scheduler,
                        }

                        self.checkpoint_fn(data)

                self.global_step += 1
                self.writer.flush()

        if self.save_dir:
            weights_save_path = os.path.join(self.save_dir, "weights.pt")
            torch.save(self.diffusion.state_dict(), weights_save_path)

        if self.checkpoint_train_end:
            self.checkpoint_train_end(
                {
                    "diffusion": self.diffusion,
                    "save_dir": self.save_dir,
                    "writer": self.writer,
                    "step": self.global_step,
                    "optimizer": self.optimizer,
                    "lr_scheduler": self.lr_scheduler,
                }
            )

        return losses
