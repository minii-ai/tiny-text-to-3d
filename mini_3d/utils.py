import matplotlib.pyplot as plt
import numpy as np
import open3d
import pyvista as pv
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


# def convert_point_cloud_to_mesh(point_cloud: torch.Tensor):
#     """
#     Params:
#         - point_cloud: (N, 3) a list of points in 3D
#     """
#     pv.set_jupyter_backend("trame")

#     if isinstance(point_cloud, torch.Tensor):
#         point_cloud = point_cloud.numpy()

#     cloud = pv.PolyData(point_cloud)
#     cloud.plot(point_size=15)

#     surf = cloud.delaunay_2d()
#     surf.plot(show_edges=True)


def convert_point_cloud_to_mesh(point_cloud: torch.Tensor) -> trimesh.Trimesh:
    """
    Params:
        - point_cloud: (N, 3) a list of points in 3D
    """

    if isinstance(point_cloud, torch.Tensor):
        point_cloud = point_cloud.numpy()

    # Convert to Open3D point cloud
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(point_cloud)

    # Estimate normals
    pcd.estimate_normals(
        # search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30)
    )

    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist

    radii = [1, 2, 4, 8]
    mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        open3d.utility.DoubleVector(radii),
        # open3d.utility.DoubleVector([radii, radii * 2]),
        # pcd, open3d.utility.DoubleVector(radii)
    )

    # Extract vertices and faces from Open3D mesh
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    # mesh = trimesh.PointCloud(vertices=vertices)
    mesh = trimesh.Trimesh(
        vertices=vertices, faces=faces, vertex_normals=np.asarray(mesh.vertex_normals)
    )

    return mesh
