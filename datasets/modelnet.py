import os

import numpy as np
import torch
import trimesh
from torch.utils.data import Dataset

PROMPT_TEMPLATES = [
    "3d render of {label}",
    "a {label}",
]


class ModelNetDataset(Dataset):
    """
    ModelNet dataset. Contains the mesh, low resolution point cloud, high resolution point cloud, label, and prompt for each item.

    Item:
    {
        "mesh": <path to mesh>,
        "label": <int>,
        "prompt": <str>,
        "low_res": <low res point cloud numpy array (N, 3)>,
        "high_res": <high res point cloud numpy array (M, 3)>
    }

    The low resolution and high resolution point cloud will be randomly resampled with probability `augment_prob`. The prompt
    will be randomly selected from `prompt_templates` with probability `augment_prob`. The base prompt for each item will
    be text corresponding to {label}.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        prompt_templates=PROMPT_TEMPLATES,
        augment_prob: float = 0.0,
    ):
        super().__init__()
        self.root = root
        self.train = train
        self.prompt_templates = prompt_templates
        self.augment_prob = augment_prob

        split = "train" if train else "test"
        off_dir = os.path.join(root, "off")
        low_res_dir = os.path.join(root, "low_res")
        high_res_dir = os.path.join(root, "high_res")

        labels = sorted(
            [
                label
                for label in os.listdir(off_dir)
                if label != ".DS_Store" and label != "README.txt"
            ]
        )

        self.labels = labels

        items = []

        for i, label in enumerate(labels):
            split_path = os.path.join(off_dir, label, split)

            for off_name in os.listdir(split_path):
                if off_name == ".DS_Store":
                    continue

                mesh_id = off_name.split(".")[0]
                off_path = os.path.join(split_path, off_name)
                low_res_path = os.path.join(low_res_dir, label, split, f"{mesh_id}.npy")
                high_res_path = os.path.join(
                    high_res_dir, label, split, f"{mesh_id}.npy"
                )

                # load low res and high res point clouds
                low_res = np.load(low_res_path)
                high_res = np.load(high_res_path)

                item = {
                    "mesh": off_path,
                    "label": i,
                    "prompt": label,
                    "low_res": low_res,
                    "high_res": high_res,
                }

                items.append(item)

        self.items = items
        self.num_low_res_points = self.items[0]["low_res"].shape[0]
        self.num_high_res_points = self.items[0]["high_res"].shape[0]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        item = {**item}

        prob = torch.rand(1).item()

        if prob <= self.augment_prob:
            # randomly get prompt from prompt_templates
            prompt = np.random.choice(self.prompt_templates).format(
                label=item["prompt"]
            )
            item["prompt"] = prompt

            # randomly resample low res and high res point cloud
            mesh = trimesh.load_mesh(item["mesh"])
            low_res_samples = np.array(
                trimesh.sample.sample_surface(mesh, self.num_low_res_points)[0].data
            )
            high_res_samples = np.array(
                trimesh.sample.sample_surface(mesh, self.num_high_res_points)[0].data
            )

            item["low_res"] = low_res_samples
            item["high_res"] = high_res_samples

        return item
