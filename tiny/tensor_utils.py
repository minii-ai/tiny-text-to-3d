import torch


def extract(arr: torch.Tensor, t: torch.Tensor, shape: torch.Size):
    """Extract value from arr at timestep t and make it broadcastable to shape."""
    B = t.shape[0]
    val = arr[t]
    return val.reshape(B, *(1 for _ in range(len(shape) - 1)))


def make_broadcastable(val: torch.Tensor, shape: torch.Size):
    """Expand val to be broadcastable to shape. Unsqueezes dimensions to the right."""
    while val.dim() < len(shape):
        val = val.unsqueeze(-1)
    return val
