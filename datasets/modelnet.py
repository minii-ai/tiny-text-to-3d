import os

import numpy as np
import open_clip
import torch
from torch.utils.data import ConcatDataset, Dataset
from tqdm import tqdm


def collate_fn_dict(batch):
    """Collate function for handling dictionaries with arbitrary keys, including string handling.

    Args:
        batch (list): A list of dictionaries. Each dictionary represents a sample from the dataset.

    Returns:
        dict: A dictionary with the same keys as the input dictionaries, where the values are batches combined appropriately.
    """
    # Initialize an empty dict to collect the batches
    collated_batch = {}

    # Each sample in the batch is a dictionary. We iterate over the keys of the first sample
    # to get all the keys assuming all samples have the same structure.
    for key in batch[0].keys():
        # Collect values from all dictionaries under the same key
        values = [sample[key] for sample in batch]

        # Check the type of the first element to decide how to combine
        if isinstance(values[0], torch.Tensor):
            # If the value is a tensor, stack them
            collated_batch[key] = torch.stack(values)
        elif isinstance(values[0], (int, float)):
            # If the value is numeric, convert to tensor
            collated_batch[key] = torch.tensor(values)
        elif isinstance(values[0], list):
            # If the value is a list, convert lists to tensor
            try:
                # Try to convert lists directly to a tensor
                collated_batch[key] = torch.tensor(values)
            except ValueError:
                # If direct conversion fails, handle as a list of tensors
                collated_batch[key] = [torch.tensor(v) for v in values]
        elif isinstance(values[0], str):
            # If the value is a string, collect into a list
            collated_batch[key] = values
        else:
            # Implement other types as needed or raise an error
            raise TypeError(
                f"Unsupported data type for batch collation: {type(values[0])}"
            )

    return collated_batch


class ModelNetDataset(Dataset):
    """
    ModelNet dataset. Contains the mesh, low resolution point cloud, high resolution point cloud, label, and prompt for each item.

    Item:
    {
        "mesh": <path to mesh>,
        "label": <int>,
        "prompt": <str>,
        "low_res": <low res point cloud tensor (N, 3)>,
        "high_res": <high res point cloud tensor (M, 3)>
    }

    The low resolution and high resolution point cloud will be randomly resampled with probability `augment_prob`. The prompt
    will be randomly selected from `prompt_templates` with probability `augment_prob`. The base prompt for each item will
    be text corresponding to {label}. To select a subset of the dataset, pass a list of labels to the `subset` argument.
    """

    @staticmethod
    def load_all(
        root: str,
        subset: list[str] = "all",
        precompute_clip_embeddings: bool = False,
        clip_model: str = "ViT-B-32",
    ):
        train_dataset = ModelNetDataset(
            root,
            train=True,
            subset=subset,
            precompute_clip_embeddings=precompute_clip_embeddings,
            clip_model=clip_model,
        )
        test_dataset = ModelNetDataset(
            root,
            train=False,
            subset=subset,
            precompute_clip_embeddings=precompute_clip_embeddings,
            clip_model=clip_model,
        )

        dataset = ConcatDataset([train_dataset, test_dataset])
        return dataset

    def __init__(
        self,
        root: str,
        train: bool = True,
        subset: list[str] = "all",
        precompute_clip_embeddings: bool = False,
        clip_model: str = "ViT-B-32",
    ):
        super().__init__()
        self.root = root
        self.train = train
        self.subset = subset

        split = "train" if train else "test"
        off_dir = os.path.join(root, "off")
        low_res_dir = os.path.join(root, "low_res")
        high_res_dir = os.path.join(root, "high_res")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if precompute_clip_embeddings:
            clip = open_clip.create_model(
                clip_model, pretrained="laion2b_s34b_b79k"
            ).to(device)
            tokenizer = open_clip.get_tokenizer(clip_model)

        labels = sorted(
            [
                label
                for label in os.listdir(off_dir)
                if label != ".DS_Store" and label != "README.txt"
            ]
        )

        self.labels = labels

        items = []

        pbar = tqdm()
        for i, label in enumerate(labels):
            if not (label in subset or subset == "all"):
                continue

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
                low_res = torch.from_numpy(np.load(low_res_path)).to(torch.float32)
                high_res = torch.from_numpy(np.load(high_res_path)).to(torch.float32)

                item = {
                    "mesh": off_path,
                    "label": i,
                    "prompt": label,
                    "low_res": low_res,
                    "high_res": high_res,
                }

                if precompute_clip_embeddings:
                    text = tokenizer(label).to(device)

                    with torch.no_grad():
                        prompt_embeds = clip.encode_text(text).cpu()
                        item["prompt_embeds"] = prompt_embeds.squeeze(0)

                items.append(item)
                pbar.update(1)

        self.items = items
        self.num_low_res_points = self.items[0]["low_res"].shape[0]
        self.num_high_res_points = self.items[0]["high_res"].shape[0]

        # get rid of clip and tokenizer after
        if precompute_clip_embeddings:
            clip.to("cpu")

            torch.cuda.empty_cache()

            del clip
            del tokenizer

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        item = {**item}

        return item
