import os

import numpy as np
import torch
from torch.utils.data import Dataset


class ModelNetPointCloudDataset(Dataset):
    def __init__(self, root: str, train: bool = True):
        super().__init__()
        self.root = root
        self.train = train

        split = "train" if train else "test"
        labels = sorted([label for label in os.listdir(root) if label != ".DS_Store"])

        self.labels = labels

        items = []

        for i, label in enumerate(labels):
            split_path = os.path.join(root, label, split)
            for point_cloud_npy in os.listdir(split_path):
                point_cloud_path = os.path.join(split_path, point_cloud_npy)
                point_cloud = np.load(point_cloud_path)
                item = (point_cloud, i)
                items.append(item)

        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        point_cloud, label = self.items[idx]
        point_cloud = torch.from_numpy(point_cloud).float()

        return point_cloud, label
