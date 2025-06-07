import os
import os.path as osp
import sys
from typing import Optional

import numpy as np
import open3d as o3d
import torch
import trimesh

import trellis.modules.sparse as sp

from ..modules.sparse import SparseTensor


def devoxelize(bin_coords: np.ndarray, resolution: int = 64):
    if bin_coords.shape[1] > 3:
        bin_coords = bin_coords[:, 1:]
    return bin_coords / resolution - 0.5


def map_slat_sp(
    ref_slat: sp.SparseTensor,
    new_slat: sp.SparseTensor,
    mask_object: Optional[o3d.geometry.TriangleMesh],
    min_bound: Optional[np.ndarray] = None,
    max_bound: Optional[np.ndarray] = None,
    transform: Optional[dict] = None,
    resolution: int = 64,
    distance_threshold: float = 0.5,  # smaller will encourage more outside points
    verbose: bool = False,
    debug: bool = False,
):
    """
    Build feats of the new slat based on the ref slat and mask object.
    Rules:
    1. Foreach point in new_slats.coords, if the point is inside the mask object, DON'T set corresponding feat, store the point index to return
    2. For other points in new_slats.coords, query the coords from ref_slats and set corresponding feat
    3. Points whose minimum distance to ref_slats is larger than distance_threshold will also be stored in mask indices

    If the mask_object_path is provided, read it with open3d for inside check, otherwise use min_bound and max_bound to get a bbox for inside check.
    """

    ref_coords, ref_feats = ref_slat.coords, ref_slat.feats.clone()
    new_coords, new_feats = new_slat.coords, new_slat.feats.clone()

    # Devoxelization, int -> float [-0.5, 0.5]
    ref_coords = devoxelize(ref_coords, resolution=resolution)
    # Reference coords should be transformed to match the new one
    ref_coords = (
        ref_coords - torch.tensor(transform["center"]).type_as(ref_coords)
    ) / (torch.tensor(transform["scale"]).type_as(ref_coords) + 1e-8)

    new_coords = devoxelize(new_coords, resolution=resolution)

    # Convert coords to numpy for processing
    new_coords_np = new_coords.cpu().numpy()

    # Create point cloud for new coordinates
    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(new_coords_np)

    if debug:
        points_all = np.concatenate((ref_coords.cpu().numpy(), new_coords_np), axis=0)
        colors_all = np.array(
            [[1.0, 0.0, 0.0]] * len(ref_coords) + [[0.0, 1.0, 0.0]] * len(new_coords)
        )
        pcd_debug = o3d.geometry.PointCloud()
        pcd_debug.points = o3d.utility.Vector3dVector(points_all)
        pcd_debug.colors = o3d.utility.Vector3dVector(colors_all)
        o3d.io.write_point_cloud("merged_pcd_refnew.ply", pcd_debug)

    # Check points inside mask object or bbox
    if mask_object is not None:
        # Load mesh and ensure it's watertight
        mesh = mask_object
        mesh.vertices = o3d.utility.Vector3dVector(
            (np.asarray(mesh.vertices) - transform["center"])
            / (transform["scale"] + 1e-8)
        )
        # o3d.io.write_triangle_mesh("mask_object.obj", mesh)
        if not mesh.is_watertight():
            print(
                "Warning: Mesh is not watertight, point-in-mesh test may be unreliable"
            )

        # Compute signed distance for all points
        scene = o3d.t.geometry.RaycastingScene()
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        _ = scene.add_triangles(mesh_t)
        query_points = o3d.core.Tensor(new_coords_np.astype(np.float32))
        signed_distances = scene.compute_signed_distance(query_points)
        inside_mask = signed_distances.numpy() < 0
        masked_point_indices = np.where(inside_mask)[0]
        outside_mask = ~inside_mask
    else:
        min_bound = (min_bound - transform["center"]) / (transform["scale"] + 1e-8)
        max_bound = (max_bound - transform["center"]) / (transform["scale"] + 1e-8)
        # Use bbox for inside check
        inside_mask = np.all(
            (new_coords_np >= min_bound) & (new_coords_np <= max_bound), axis=1
        )
        masked_point_indices = np.where(inside_mask)[0]
        outside_mask = ~inside_mask

    # Initialize far_points_mask
    far_points_mask = torch.zeros(
        len(new_coords), dtype=torch.bool, device=new_coords.device
    )

    # Process points outside mask using vectorized distance computation
    if np.any(outside_mask):
        outside_points = new_coords[torch.tensor(outside_mask).bool().cuda()].clone()
        # Compute pairwise distances between outside points and reference points
        # Using broadcasting for vectorized computation
        # Shape: (n_outside_points, n_ref_points)
        dists = torch.sum((outside_points[:, None] - ref_coords[None, :]) ** 2, dim=2)
        min_dists, closest_ref_indices = torch.min(dists, dim=1)

        # Find points with large minimum distances
        far_points_mask_outside = min_dists > distance_threshold
        far_points_mask[
            torch.where(torch.tensor(outside_mask).bool().cuda())[0][
                far_points_mask_outside
            ]
        ] = True

        close_points_mask = ~far_points_mask_outside
        # Add indices of far points to masked_point_indices
        outside_indices = torch.where(torch.tensor(outside_mask).bool())[0].cuda()
        far_point_indices = outside_indices[far_points_mask_outside]
        masked_point_indices = np.concatenate(
            [masked_point_indices, far_point_indices.cpu().numpy()]
        )

        # Update features only for close points
        close_outside_indices = outside_indices[close_points_mask]
        close_ref_indices = closest_ref_indices[close_points_mask]
        new_feats[close_outside_indices] = ref_feats[close_ref_indices].clone()

    # Convert arrays back to tensors with proper device placement
    masked_point_indices = torch.tensor(
        masked_point_indices, dtype=torch.long, device=new_coords.device
    )

    new_coords = torch.cat(
        (torch.zeros((len(new_coords), 1)).type_as(new_coords), new_coords), dim=1
    )

    # Create output slat with updated features
    output_slat = SparseTensor(
        coords=(resolution * (new_coords + 0.5)).int(),  # Already a tensor
        feats=new_feats,  # Already a tensor
    )

    if verbose:
        print("Output slat shape:", output_slat.coords.shape, output_slat.feats.shape)
        print("Masked point indices shape:", masked_point_indices.shape)
        print(
            "Number of points masked by distance threshold:",
            torch.sum(far_points_mask).item() if np.any(outside_mask) else 0,
        )

    if debug:
        # Visualization for debugging
        # Create point clouds for different categories
        inside_pcd = o3d.geometry.PointCloud()
        outside_close_pcd = o3d.geometry.PointCloud()
        far_pcd = o3d.geometry.PointCloud()

        # Convert masks to numpy for indexing
        inside_mask_np = inside_mask
        outside_mask_np = outside_mask
        far_points_mask_np = far_points_mask.cpu().numpy()

        # Assign points to different categories
        inside_points = new_coords_np[inside_mask_np]
        outside_close_points = new_coords_np[outside_mask_np & ~far_points_mask_np]
        far_points = new_coords_np[far_points_mask_np]

        # Set points and colors for each category
        inside_pcd.points = o3d.utility.Vector3dVector(inside_points)
        outside_close_pcd.points = o3d.utility.Vector3dVector(outside_close_points)
        far_pcd.points = o3d.utility.Vector3dVector(far_points)

        # Set colors: red for inside, green for outside close, blue for far points
        inside_pcd.paint_uniform_color([1, 0, 0])  # Red
        outside_close_pcd.paint_uniform_color([0, 1, 0])  # Green
        far_pcd.paint_uniform_color([0, 0, 1])  # Blue

        # save the colorized point cloud for debug
        merged_points = np.concatenate(
            (inside_points, outside_close_points, far_points), axis=0
        )
        merged_colors = np.array(
            (
                [[1, 0, 0]] * len(inside_points)
                + [[0, 1, 0]] * len(outside_close_points)
                + [[0, 0, 1]] * len(far_points)
            )
        ).reshape(-1, 3)
        merged_pcd = o3d.geometry.PointCloud()
        merged_pcd.points = o3d.utility.Vector3dVector(merged_points)
        merged_pcd.colors = o3d.utility.Vector3dVector(merged_colors)
        o3d.io.write_point_cloud("merged_pcd.ply", merged_pcd)

    # should return which feats we should repaint, marked as 1.0
    mask_feats = torch.zeros(
        (output_slat.feats.shape[0], 1),
    ).type_as(output_slat.feats)
    mask_feats[masked_point_indices] = 1.0
    return output_slat, mask_feats
