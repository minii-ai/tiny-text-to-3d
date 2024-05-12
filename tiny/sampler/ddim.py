import torch
import torch.nn.functional as F

from ..noise_schedule import NoiseScheduler
from ..tensor_utils import extract, make_broadcastable
from .ddpm import DDPMSampler


class DDIMSampler(DDPMSampler):
    def __init__(self, noise_scheduler: NoiseScheduler, eta: float = 0.0):
        super().__init__(noise_scheduler)
        self.eta = eta

        # divide timesteps up uniformly
        self._num_timesteps = noise_scheduler.num_timesteps
        self.set_timesteps(noise_scheduler.num_timesteps)

    def set_timesteps(self, num_timesteps: int):
        assert (
            num_timesteps <= self.noise_scheduler.num_timesteps
        ), "Number of timesteps has to be less than the number of timesteps in the noise schedule."

        if num_timesteps == self._num_timesteps:
            return

        timesteps = torch.linspace(
            0, self.noise_scheduler.num_timesteps - 1, num_timesteps, device=self.device
        ).int()

        alphas_cumprod = self.noise_scheduler.alphas_cumprod[timesteps]
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], pad=(1, 0), value=1)

        sqrt_alphas_cumprod = self.noise_scheduler.sqrt_alphas_cumprod[timesteps]
        sqrt_alphas_cumprod_prev = F.pad(sqrt_alphas_cumprod[:-1], pad=(1, 0), value=1)
        sigmas_squared = self.eta * self.q_posterior_variance[timesteps]
        sqrt_one_minus_alphas_cumprod_prev_minus_sigmas_squared = torch.sqrt(
            1 - alphas_cumprod_prev - sigmas_squared
        )

        self.register_buffer("timesteps", timesteps)
        self.register_buffer("sqrt_alphas_cumprod_prev", sqrt_alphas_cumprod_prev)
        self.register_buffer("sigmas_squared", sigmas_squared)
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod_prev_minus_sigmas_squared",
            sqrt_one_minus_alphas_cumprod_prev_minus_sigmas_squared,
        )

        self._num_timesteps = num_timesteps

    def scale_timesteps(self, t: torch.Tensor):
        return (t / (self.noise_scheduler.num_timesteps / self._num_timesteps)).int()

    def p_mean_variance_ddim(
        self,
        model,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond=None,
        guidance_scale: float = 1.0,
        clip_denoised: bool = False,
        use_cfg: bool = False,
    ):
        # scale t so that indices match
        t_scaled = self.scale_timesteps(t)

        model_pred = self.model_prediction(
            model,
            x_t=x_t,
            t=t,
            cond=cond,
            guidance_scale=guidance_scale,
            clip_denoised=clip_denoised,
            use_cfg=use_cfg,
        )
        pred_noise, pred_x_start = model_pred["pred_noise"], model_pred["pred_x_start"]

        mean = (
            extract(self.sqrt_alphas_cumprod_prev, t_scaled, x_t.shape) * pred_x_start
            + extract(
                self.sqrt_one_minus_alphas_cumprod_prev_minus_sigmas_squared,
                t_scaled,
                x_t.shape,
            )
            * pred_noise
        )
        variance = extract(self.sigmas_squared, t_scaled, x_t.shape)

        return mean, variance

    def step(
        self,
        model,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond=None,
        guidance_scale: float = 1.0,
        clip_denoised: bool = False,
        use_cfg: bool = False,
    ):
        mean, variance = self.p_mean_variance_ddim(
            model,
            x_t=x_t,
            t=t,
            cond=cond,
            guidance_scale=guidance_scale,
            clip_denoised=clip_denoised,
            use_cfg=use_cfg,
        )

        # zero out variance at timestep t = 0
        nonzero_mask = make_broadcastable(t != 0, variance.shape)

        # sample
        eps = torch.randn_like(x_t)
        prev_x = mean + variance**0.5 * nonzero_mask * eps

        return prev_x
