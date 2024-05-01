import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import trimesh
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from torchvision.transforms import v2


def visualize_point_cloud(point_cloud: torch.Tensor):
    """
    Params:
        - point_cloud: (N, D) a list of points in 2D or 3D space
    """
    fig = plt.figure()
    dim = point_cloud.shape[1]

    assert dim in {2, 3}

    if dim == 2:
        ax = fig.add_subplot(111)
        X, Y = point_cloud[:, 0], point_cloud[:, 1]
        ax.scatter(X, Y)
    elif dim == 3:
        ax = fig.add_subplot(111, projection="3d")
        X, Y, Z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
        ax.scatter(X, Y, Z)

    plt.show()


def convert_point_cloud_to_ply(point_cloud: torch.Tensor, save_path: str):
    if isinstance(point_cloud, torch.Tensor):
        point_cloud = point_cloud.numpy()

    point_cloud = trimesh.points.PointCloud(point_cloud)
    point_cloud.export(save_path)


def to_pil_image(x: torch.Tensor):
    # normalize between 0 and 1
    x = ((x + 1) / 2).clamp(0, 1)
    return v2.ToPILImage()(x)


def plot_images(images: list[Image.Image], **kwargs):
    rows = 1
    cols = len(images)

    fig, axs = plt.subplots(figsize=(25, 25), nrows=rows, ncols=cols, squeeze=True)

    if len(images) == 1:
        axs = [axs]

    for i, img in enumerate(images):
        axs[i].imshow(img, **kwargs)
        axs[i].axis("off")

    plt.tight_layout()


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters())
