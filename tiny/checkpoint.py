import os

from tqdm import tqdm

from .utils import create_gif_from_pil, plot_point_clouds


class LogSamples:
    def __init__(
        self,
        num_samples: int,
        dir: str,
        rows: int,
        cols: int,
        clip_denoised: bool = True,
        model_kwargs={},
    ):
        self.num_samples = num_samples
        self.dir = dir
        self.rows = rows
        self.cols = cols
        self.clip_denoised = clip_denoised
        self.model_kwargs = model_kwargs

    def __call__(self, data):
        ddpm = data["ddpm"]
        save_dir = data["save_dir"]
        epoch = data["epoch"]

        samples = ddpm.sample(
            batch_size=self.num_samples,
            clip_denoised=self.clip_denoised,
            **self.model_kwargs,
        )

        plot = plot_point_clouds(samples, self.rows, self.cols)

        # save plot
        save_dir = os.path.join(save_dir, self.dir)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"samples_{epoch}.png")
        plot.save(save_path)


class LogDiffusionGIF:
    def __init__(self, num_samples: int, rows: int, cols: int, save_dir: str, fps: int):
        self.num_samples = num_samples
        self.rows = rows
        self.cols = cols
        self.save_dir = save_dir
        self.fps = fps

    def __call__(self, data):
        ddpm = data["ddpm"]
        samples = []
        for sample in ddpm.sample_progressive(
            batch_size=self.num_samples, clip_denoised=True
        ):
            samples.append(sample)

        plots = [
            plot_point_clouds(sample, self.rows, self.cols) for sample in tqdm(samples)
        ]

        save_path = os.path.join(self.save_dir, "diffusion.gif")
        create_gif_from_pil(plots, save_path, self.fps)
