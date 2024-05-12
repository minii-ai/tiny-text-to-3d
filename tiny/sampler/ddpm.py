import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..noise_schedule import NoiseScheduler
from ..tensor_utils import extract, make_broadcastable
from .base import Sampler


class DDPMSampler(Sampler):
    def __init__(self, noise_scheduler: NoiseScheduler):
        super().__init__(noise_scheduler)

        betas = self.noise_scheduler.betas
        alphas = self.noise_scheduler.alphas
        alphas_cumprod = self.noise_scheduler.alphas_cumprod

        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], pad=(1, 0), value=1)
        q_posterior_mean_coef1 = (torch.sqrt(alphas_cumprod_prev) * betas) / (
            1 - alphas_cumprod
        )
        q_posterior_mean_coef2 = (torch.sqrt(alphas) * (1 - alphas_cumprod_prev)) / (
            1 - alphas_cumprod
        )
        q_posterior_variance = (1 - alphas_cumprod_prev) * betas / (1 - alphas_cumprod)

        self.register_buffer("q_posterior_mean_coef1", q_posterior_mean_coef1)
        self.register_buffer("q_posterior_mean_coef2", q_posterior_mean_coef2)
        self.register_buffer("q_posterior_variance", q_posterior_variance)

    @property
    def device(self):
        return self.q_posterior_mean_coef1.device

    def set_timesteps(self, num_timesteps: int):
        # for DDPMs, the number of timesteps is fixed to the number of timesteps in the noise schedule
        pass

    def predict_x_start(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        sqrt_one_minus_alphas_cumprod = (
            self.noise_scheduler.sqrt_one_minus_alphas_cumprod
        )
        sqrt_alphas_cumprod = self.noise_scheduler.sqrt_alphas_cumprod

        x_start = (
            x_t - extract(sqrt_one_minus_alphas_cumprod, t, x_t.shape) * noise
        ) / extract(sqrt_alphas_cumprod, t, x_t.shape)

        return x_start

    def q_posterior_mean_variance(
        self, x_t: torch.Tensor, x_start: torch.Tensor, t: torch.Tensor
    ):
        """
        Gets the mean and variance of q(x_{t-1} | x_t, x_start)
        """

        mean = (
            extract(self.q_posterior_mean_coef1, t, x_start.shape) * x_start
            + extract(self.q_posterior_mean_coef2, t, x_t.shape) * x_t
        )
        variance = extract(self.q_posterior_variance, t, x_t.shape)

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
        """
        Samples x_{t-1} ~ p(x_{t-1} | x_t, t) or p(x_{t-1} | x_t, t, c) if cond is provided.
        """

        if use_cfg:
            # predict unconditional and conditional noise
            pred_cond_noise = model(x_t, t, cond)
            pred_uncond_noise = model(x_t, t)

            # classifier-free guidance
            pred_noise = (
                1 - guidance_scale
            ) * pred_uncond_noise + guidance_scale * pred_cond_noise
        else:
            if cond is not None:
                pred_noise = model(x_t, t, cond)
            else:
                pred_noise = model(x_t, t)

        # predict x_start
        pred_x_start = self.predict_x_start(x_t, t, pred_noise)

        if clip_denoised:
            pred_x_start = pred_x_start.clip(-1.0, 1.0)

        # get mean and variance from q(x_{t-1} | x_t, pred_x_start)
        mean, variance = self.q_posterior_mean_variance(x_t, pred_x_start, t)

        # zero out variance at timestep t = 0
        nonzero_mask = make_broadcastable(t != 0, variance.shape)

        # sample
        eps = torch.randn_like(x_t)
        prev_x = mean + variance**0.5 * nonzero_mask * eps

        return prev_x

    # @torch.no_grad()
    # def sample_loop(
    #     self,
    #     model,
    #     shape: tuple,
    #     cond: torch.Tensor = None,
    #     num_inference_steps: int = None,
    #     clip_denoised: bool = False,
    #     guidance_scale: float = 1.0,
    #     use_cfg: bool = False,
    # ):
    #     for sample in self.sample_loop_progressive(
    #         model,
    #         shape,
    #         cond=cond,
    #         clip_denoised=clip_denoised,
    #         num_inference_steps=num_inference_steps,
    #         guidance_scale=guidance_scale,
    #         use_cfg=use_cfg,
    #     ):
    #         pass

    #     return sample

    # def sample_loop_progressive(
    #     self,
    #     model,
    #     shape,
    #     cond: torch.Tensor = None,
    #     num_inference_steps: int = None,
    #     guidance_scale: float = 1.0,
    #     clip_denoised: bool = False,
    #     use_cfg: bool = False,
    # ):
    #     """
    #     Yields sample at each timestep during the sampling loop
    #     """

    #     B = shape[0]

    #     # start with random gaussian noise
    #     x_t = torch.randn(*shape, device=self.device)
    #     timesteps = range(self.noise_scheduler.num_timesteps)[::-1]
    #     cond = model.prepare_cond(cond)

    #     # denoise progressively at each timestep
    #     for t in tqdm(timesteps):
    #         t = torch.full((B,), t, device=self.device)
    #         x_t = self.step(
    #             model,
    #             x_t,
    #             t,
    #             cond=cond,
    #             guidance_scale=guidance_scale,
    #             clip_denoised=clip_denoised,
    #             use_cfg=use_cfg,
    #         )
    #         yield x_t
