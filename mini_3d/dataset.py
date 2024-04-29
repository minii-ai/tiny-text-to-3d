from torch.utils.data import Dataset


class ModelNetDataset(Dataset):
    def __init__(self, root: str, train: bool = True):
        super().__init__()

    def __len__(self):
        pass

    def __getitem__(self, idx: int):
        pass
