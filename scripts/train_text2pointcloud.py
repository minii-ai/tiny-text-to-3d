import argparse
import json
import sys

import torch
from torch.utils.data import DataLoader

sys.path.append("../")
from datasets import ModelNetDataset, collate_fn_dict
from tiny import PointCloudDiffusion, PointCloudDiffusionTrainer


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

    return parser.parse_args()


def read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


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

    trainer = PointCloudDiffusionTrainer(
        diffusion=diffusion,
        train_loader=dataloader,
        lr=train_config["lr"],
        num_epochs=train_config["num_epochs"],
        get_batch_fn=get_batch_fn,
        save_dir=args.save_dir,
        device=args.device,
    )

    trainer.train()


if __name__ == "__main__":
    args = parse_args()
    main(args)
