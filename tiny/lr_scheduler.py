import math

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler


class CosineAnnealingWithWarmupLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        init_lr: float,
        warmup_steps: int,
        T_max: int,
        eta_min: float,
        last_epoch: int = -1,
    ):
        for param_group in optimizer.param_groups:
            lr = param_group["lr"]

        self._lr = lr
        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.cosine_annealing = CosineAnnealingLR(
            optimizer, T_max - warmup_steps, eta_min, last_epoch
        )

        super().__init__(optimizer, last_epoch)

    def step(self, *args):
        if self._step_count <= self.warmup_steps:
            values = self.get_lr()
            for param_group, lr in zip(
                self.cosine_annealing.optimizer.param_groups, values
            ):
                param_group["lr"] = lr
        else:
            self.cosine_annealing.step(*args)

        self._step_count += 1

    def get_lr(self):
        if self._step_count <= self.warmup_steps:
            lr = [
                ((self._lr - self.init_lr) / (self.warmup_steps - 1))
                * (self._step_count - 1)
                + self.init_lr
            ]

            return lr
        else:
            return self.cosine_annealing.get_last_lr()
