from typing import Literal

import torch
import torch.nn as nn

ScheduleType = Literal["linear", "cosine"]


class NoiseScheduler(nn.Module):
    def __init__(self, num_timesteps: int, schedule_type: ScheduleType = "cosine"):
        super().__init__()

    def add_noise(
        self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None
    ):
        """
        Adds noise to x_start for t timesteps. Samples x_t ~ q(x_t | x_start).
        """
