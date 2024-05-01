from typing import TypedDict, Union

import torch
from torch.utils.data import Dataset


class DatasetItem(TypedDict):
    data: torch.Tensor
    label: Union[int, None]
    text: Union[torch.Tensor, int, str, None]


class BaseDataset(Dataset):
    @staticmethod
    def collate(items: list[DatasetItem]):
        data = torch.stack([item["data"] for item in items], dim=0)
        batch = {"data": data}

        if items[0].get("label", None) is not None:
            labels = torch.tensor(
                [item["label"] for item in items], dtype=torch.float32
            )
            batch["labels"] = labels

        if items[0].get("text", None) is not None:
            texts = [item["text"] for item in items]
            batch["texts"] = texts

        return batch

    def __getitem__(self, idx: int) -> DatasetItem:
        raise NotImplementedError
