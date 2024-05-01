import argparse
import json
import sys

from torch.utils.data import DataLoader

sys.path.append("../")

from datasets import SineCosineDataset
from tiny import PointCloudDDPM, PointCloudDiffusionTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config", type=str, required=True)
    parser.add_argument("--ddpm_config", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    return parser.parse_args()


def read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def main(args):
    train_config = read_json(args.train_config)
    ddpm_config = read_json(args.ddpm_config)

    # prepare dataset
    dataset = SineCosineDataset(type="both")
    dataloader = DataLoader(
        dataset, batch_size=train_config["batch_size"], shuffle=True
    )

    # prepare model
    ddpm = PointCloudDDPM.from_config(ddpm_config)

    # train model
    trainer = PointCloudDiffusionTrainer(
        ddpm,
        dataloader,
        model_type="uncond",
        lr=train_config["lr"],
        num_epochs=train_config["num_epochs"],
        save_dir=args.save_dir,
    )

    trainer.train()


if __name__ == "__main__":
    args = parse_args()
    main(args)
