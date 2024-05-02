import io

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import trimesh
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from torchvision.transforms import v2


def _fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def plot_point_cloud(point_cloud: torch.Tensor):
    """
    Params:
        - point_cloud: (N, D) a list of points in 2D or 3D space
    """
    fig = plt.figure()
    dim = point_cloud.shape[1]
    point_cloud = point_cloud.cpu()

    assert dim in {2, 3}

    if dim == 2:
        ax = fig.add_subplot(111)
        X, Y = point_cloud[:, 0], point_cloud[:, 1]
        ax.scatter(X, Y)
    elif dim == 3:
        ax = fig.add_subplot(111, projection="3d")
        X, Y, Z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
        ax.scatter(X, Y, Z)

    return _fig_to_image(fig)


def plot_point_clouds(point_clouds: torch.Tensor, rows: int, cols: int):
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.ravel()
    dim = point_clouds.shape[2]
    point_clouds = point_clouds.cpu()

    for idx, ax in enumerate(axes):
        if idx < len(point_clouds):

            if dim == 2:
                X, Y = point_clouds[idx, :, 0], point_clouds[idx, :, 1]
                ax.scatter(X, Y)
            elif dim == 3:
                X, Y, Z = (
                    point_clouds[idx, :, 0],
                    point_clouds[idx, :, 1],
                    point_clouds[idx, :, 2],
                )
                ax.scatter(X, Y, Z)

            ax.axis("equal")  # Ensures that the scale of x and y axes are the same
        else:
            ax.axis("off")  # Turn off axis if there's no data to plot

    return _fig_to_image(fig)


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


def create_video_from_pil(images: list[Image.Image], video_name: str, fps: int):
    # Assuming all images are the same size, get dimensions from the first image
    width, height = images[0].size

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use 'mp4v' for MP4
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for pil_image in images:
        # Convert the PIL image to an OpenCV image (convert RGB to BGR)
        open_cv_image = np.array(pil_image.convert("RGB"))
        video_frame = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

        # Write the frame to the video
        video.write(video_frame)

    # Release the video writer
    video.release()


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters())
