import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append("../")
from datasets import ModelNetDataset, collate_fn_dict
from tiny import PointCloudDiffusion, PointCloudDiffusionTrainer
from tiny.utils import plot_point_clouds, to_tensor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--augment_prob", type=float, default=0.0)
    parser.add_argument("--subset", type=str, default="all")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument(
        "--diffusion_config",
        type=str,
        default="../configs/diffusion/text2pointcloud_tiny.json",
    )
    parser.add_argument(
        "--train_config", type=str, default="../configs/train/text2pointcloud_tiny.json"
    )
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--checkpoint_every", type=int, default=5)

    return parser.parse_args()


def read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


from torch.utils.data import Subset


def main(args):
    torch.manual_seed(0)

    train_config = read_json(args.train_config)
    batch_size = train_config["batch_size"]

    # prepare dataset and dataloader
    if args.subset == "all":
        subset = "all"
    else:
        subset = args.subset.split(",")

    dataset = ModelNetDataset.load_all(
        args.dataset_dir, augment_prob=args.augment_prob, subset=subset
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn_dict,
    )

    # prepare diffusion model
    diffusion_config = read_json(args.diffusion_config)
    diffusion = PointCloudDiffusion.from_config(diffusion_config)

    # create trainer
    def get_batch_fn(batch):
        return {"data": batch["low_res"], "cond": batch["prompt"]}

    def checkpoint_fn(data):
        epoch = data["epoch"]
        writer = data["writer"]
        global_step = data["global_step"]
        save_dir = args.save_dir

        # create checkpoint dir
        checkpoint_dir = os.path.join(save_dir, f"checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # generate point clouds for each prompt
        diffusion = data["diffusion"]
        eval_config = train_config["eval"]
        prompts = eval_config["prompts"]
        guidance_scale = eval_config["guidance_scale"]
        num_inference_steps = eval_config["num_inference_steps"]
        num_samples = eval_config["num_samples"]

        point_cloud_samples = []
        titles = []

        for prompt in prompts:
            cond = [prompt] * num_samples
            point_clouds = diffusion.sample_loop(
                batch_size=num_samples,
                cond=cond,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                use_cfg=True,
            )

            titles += cond
            point_cloud_samples.append(point_clouds.cpu())

        point_cloud_samples = torch.cat(point_cloud_samples, dim=0)
        image = plot_point_clouds(
            point_cloud_samples, len(prompts), num_samples, titles
        ).convert("RGB")

        image = to_tensor(image)

        writer.add_image("samples", image, global_step)
        weights_path = os.path.join(checkpoint_dir, f"weights_{epoch}.pt")
        torch.save(diffusion.state_dict(), weights_path)

    trainer = PointCloudDiffusionTrainer(
        diffusion=diffusion,
        train_loader=dataloader,
        lr=train_config["lr"],
        num_epochs=train_config["num_epochs"],
        checkpoint_every=train_config["eval"]["checkpoint_every"],
        get_batch_fn=get_batch_fn,
        checkpoint_fn=checkpoint_fn,
        save_dir=args.save_dir,
        device=args.device,
    )

    trainer.train()


if __name__ == "__main__":
    args = parse_args()
    main(args)
