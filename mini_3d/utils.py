import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from mpl_toolkits.mplot3d import Axes3D


def visualize_point_cloud(point_cloud: torch.Tensor):
    """
    Params:
        - point_cloud: (N, 3) a list of points in 3D
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    X, Y, Z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
    ax.scatter(X, Y, Z)

    plt.show()


def convert_point_cloud_to_ply(point_cloud: torch.Tensor, save_path: str):
    if isinstance(point_cloud, torch.Tensor):
        point_cloud = point_cloud.numpy()

    point_cloud = trimesh.points.PointCloud(point_cloud)
    point_cloud.export(save_path)
