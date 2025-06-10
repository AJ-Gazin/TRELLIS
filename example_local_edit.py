#!/usr/bin/env python
# coding=utf-8

import os
import sys

os.environ["ATTN_BACKEND"] = (
    "xformers"  # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
)
os.environ["SPCONV_ALGO"] = "native"  # Can be 'native' or 'auto', default is 'auto'.
# 'auto' is faster but will do benchmarking at the beginning.
# Recommended to set to 'native' if run only once.

import imageio
import numpy as np
import open3d as o3d
import torch
import trimesh
from PIL import Image
from tqdm import tqdm, trange

from trellis import models
from trellis.modules.sparse import SparseTensor, sparse_cat
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils, render_utils

"""
Image-conditioned local editing (Sec3.4 of the paper)
"""


def bake_scene_to_mesh(mesh_path: str):
    mesh: trimesh.Scene = trimesh.load(mesh_path)

    geo_name = list(mesh.geometry.keys())[0]
    geo = mesh.geometry[geo_name]
    nodes = mesh.graph.geometry_nodes[geo_name]
    transform = mesh.graph[nodes[0]][0]
    geo.apply_transform(transform)
    return geo


def create_o3d_mesh(mesh: trimesh.Trimesh):
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices),
        o3d.utility.Vector3iVector(mesh.faces),
    )
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    return mesh


def load_sparse_structure(ss_path: str, get_tensor: bool = False):
    ss_np = np.load(ss_path)["mean"]
    if get_tensor:
        return torch.from_numpy(ss_np).float().cuda()[None]
    return ss_np


def load_slat(slat_path: str):
    slat = np.load(slat_path)
    coords, feats = slat["coords"], slat["feats"]
    coords = torch.from_numpy(coords).int().cuda()
    feats = torch.from_numpy(feats).float().cuda()
    return SparseTensor(coords=coords, feats=feats)


def load_ss_and_slat(indir: str):
    ss_path = os.path.join(indir, "ss.npz")
    ss = load_sparse_structure(ss_path, get_tensor=True)
    slat_path = os.path.join(indir, "slat.npz")
    slat = load_slat(slat_path)
    return ss, slat


def voxelize(mesh: o3d.geometry.TriangleMesh, resolution: int = 64):
    # mesh = o3d.io.read_triangle_mesh(mesh_path)
    # clamp vertices to the range [-0.5, 0.5]
    vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh,
        voxel_size=1 / resolution,
        min_bound=(-0.5, -0.5, -0.5),
        max_bound=(0.5, 0.5, 0.5),
    )
    vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    binary_voxel = np.zeros((resolution, resolution, resolution), dtype=bool)
    binary_voxel[vertices[:, 0], vertices[:, 1], vertices[:, 2]] = True
    return binary_voxel


def binarize_voxels(positions: np.ndarray, resolution: int = 64):
    coords = ((torch.tensor(positions) + 0.5) * resolution).int().contiguous()
    ss = torch.zeros(1, resolution, resolution, resolution, dtype=torch.long)
    ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    return ss


def voxelize_and_genmask(
    mesh_path: str, mask_path: str, mesh_resolution: int = 64, mask_resolution: int = 16
):
    mesh: o3d.geometry.TriangleMesh = create_o3d_mesh(bake_scene_to_mesh(mesh_path))
    mask: o3d.geometry.TriangleMesh = create_o3d_mesh(bake_scene_to_mesh(mask_path))
    mesh_v = np.asarray(mesh.vertices)
    mask_v = np.asarray(mask.vertices)
    # re-normalise both of them to [-0.5, 0.5]
    min_bound_mesh = mesh_v.min(axis=0)
    max_bound_mesh = mesh_v.max(axis=0)
    min_bound_mask = mask_v.min(axis=0)
    max_bound_mask = mask_v.max(axis=0)
    min_bound = np.minimum(min_bound_mesh, min_bound_mask)
    max_bound = np.maximum(max_bound_mesh, max_bound_mask)
    center = (min_bound + max_bound) / 2
    scale = (max_bound - min_bound).max()
    mesh.vertices = o3d.utility.Vector3dVector(
        (np.asarray(mesh.vertices) - center) / (scale + 1e-8)
    )
    mask.vertices = o3d.utility.Vector3dVector(
        (np.asarray(mask.vertices) - center) / (scale + 1e-8)
    )
    # o3d.io.write_triangle_mesh("debug_mesh.obj", mesh)
    # o3d.io.write_triangle_mesh("debug_mask.obj", mask)
    # print(np.asarray(mesh.vertices).min(axis=0), np.asarray(mesh.vertices).max(axis=0))
    # print(np.asarray(mask.vertices).min(axis=0), np.asarray(mask.vertices).max(axis=0))
    return (
        (center, scale),
        voxelize(mesh, mesh_resolution),
        voxelize(mask, mesh_resolution),
        voxelize(mask, mask_resolution),
    )


feat_model = "dinov2_vitl14_reg"
enc_pretrained = "microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16"
latent_name = f"{feat_model}_{enc_pretrained.split('/')[-1]}"
# the encoder from voxel to sparse structure latent
encoder = models.from_pretrained(enc_pretrained).eval().cuda()

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.cuda()

# Test image path and saving directory
saveroot = "results/local_edit"

example_pairs = [
    (
        "doll_head_lantern",
        "./assets/example_local_edit/doll2/latents/",
        "./assets/example_local_edit/doll2/doll.glb",
        "./assets/example_local_edit/doll2/head_mask.glb",
        "./assets/example_local_edit/doll2/head_lantern.png",
    ),
    (
        "doll_headcenter_lantern",
        "./assets/example_local_edit/doll2/latents/",
        "./assets/example_local_edit/doll2/doll.glb",
        "./assets/example_local_edit/doll2/headcenter_mask.glb",
        "./assets/example_local_edit/doll2/head_lantern.png",
    ),
    (
        "doll_headcenter_furry",
        "./assets/example_local_edit/doll2/latents/",
        "./assets/example_local_edit/doll2/doll.glb",
        "./assets/example_local_edit/doll2/headcenter_mask.glb",
        "./assets/example_local_edit/doll2/head_furry.png",
    ),
    (
        "doll_headcenter_mailbox",
        "./assets/example_local_edit/doll2/latents/",
        "./assets/example_local_edit/doll2/doll.glb",
        "./assets/example_local_edit/doll2/headcenter_mask.glb",
        "./assets/example_local_edit/doll2/head_mailbox.png",
    ),
][:3]


def run_experiments(
    encoded, slat, ss_mask2, image, mask_object, center_and_scale, savedir
):
    # Define different combinations of parameters to test
    # cfg_strengths = [5.0, 7.5, 10.0]
    # resample_times_list = [4, 6, 8]

    steps = [12, 20, 35, 50]
    cfg_strengths = [(5.0, 3.0), (7.5, 5.0), (10.0, 7.5)]
    resample_times_list = [1, 2, 3, 5]
    resample_methods = [1]

    steps = [12, 25]
    cfg_strengths = [(5.0, 3.0), (7.5, 5.0)]

    for step in steps:
        for cfg_strength in cfg_strengths:
            cfg_ss, cfg_slat = cfg_strength
            for resample_times in resample_times_list:
                for resample_method in resample_methods:
                    # Create a subdirectory for this combination
                    experiment_dir = os.path.join(
                        savedir,
                        f"cfg_{cfg_ss:.1f}_{cfg_slat:.1f}_resample{resample_times}_steps{step}",
                    )
                    if os.path.exists(os.path.join(experiment_dir, "edited.glb")):
                        print("Already find the extracted mesh file, continue...")
                        continue

                    if os.path.exists(experiment_dir):
                        print("Already find the experiment dir, continue...")
                        continue
                    try:
                        outputs_all = pipeline.run_local_editing(
                            encoded,
                            slat,
                            ss_mask2,
                            image,
                            seed=1,
                            mask_object=mask_object,
                            post_mask_transform={
                                "center": center_and_scale[0],
                                "scale": center_and_scale[1],
                            },
                            sparse_structure_sampler_params={
                                "steps": step,
                                "cfg_strength": cfg_ss,
                                "resample_times": resample_times,
                                "resample_method": 1,
                                "rescale_t": 3.0,
                            },
                            # resample_method = 1 -> terms x_t as x_0 in forward diffusion
                            slat_sampler_params={
                                "steps": step,
                                "cfg_strength": cfg_slat,
                                # "resample_times": resample_times,  # = 1 disables resampling
                                # "resample_method": resample_method,
                                # "rescale_t": 3.0,
                            },
                            enable_slat_repaint=False,
                            return_all=True,
                        )
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        print(
                            f"Error: {e} during running the {step} {cfg_ss:.1f}_{cfg_slat:.1f} {resample_times} {resample_method}"
                        )
                        continue

                    torch.cuda.empty_cache()
                    # save the sparse structure and slats
                    outputs = outputs_all["outputs"]

                    os.makedirs(experiment_dir, exist_ok=True)
                    os.makedirs(os.path.join(experiment_dir, "latents"), exist_ok=True)

                    # Save outputs
                    np.savez(
                        os.path.join(experiment_dir, "latents", "ss.npz"),
                        mean=outputs_all["ss_coords"],
                    )
                    np.savez(
                        os.path.join(experiment_dir, "latents", "slat.npz"),
                        coords=outputs_all["slat_coords"],
                        feats=outputs_all["slat_feats"],
                    )

                    # Render the outputs
                    video = render_utils.render_video(outputs["gaussian"][0])["color"]
                    imageio.mimsave(
                        os.path.join(experiment_dir, "sample_gs.mp4"), video, fps=30
                    )
                    video = render_utils.render_video(outputs["radiance_field"][0])[
                        "color"
                    ]
                    imageio.mimsave(
                        os.path.join(experiment_dir, "sample_rf.mp4"), video, fps=30
                    )
                    video = render_utils.render_video(outputs["mesh"][0])["normal"]
                    imageio.mimsave(
                        os.path.join(experiment_dir, "sample_mesh.mp4"), video, fps=30
                    )
                    try:
                        # GLB files can be extracted from the outputs
                        glb = postprocessing_utils.to_trimesh(
                            outputs["radiance_field"][0],
                            outputs["mesh"][0],
                            # Optional parameters
                            texture_bake_mode="fast",  # Low VRAM mode defaults baking to fast mode
                            # Postprocessing methods
                            postprocess_mode="simplify",
                            remesh_iters=10,
                            subdivision_times=1,
                            simplify=0.95,  # Ratio of triangles to remove in the simplification process
                            texture_size=1024,  # Size of the texture used for the GLB
                            debug=False,
                            verbose=True,
                        )
                        glb.export(os.path.join(experiment_dir, "edited.glb"))
                    except:
                        pass


for example in example_pairs:
    instance_name, latent_dir, mesh_path, mask_path, image_path = example
    savedir = os.path.join(saveroot, instance_name + "_ssonly")
    os.makedirs(savedir, exist_ok=True)
    ss, slat = load_ss_and_slat(latent_dir)
    mask_object = create_o3d_mesh(bake_scene_to_mesh(mask_path))
    # can also binarize the `sparse_structure` as the input
    # ss_mask is voxelized under the `mesh_resolution`
    # while ss_mask2 is voxelized under the `mask_resolution`
    center_and_scale, binary_voxels, ss_mask, ss_mask2 = voxelize_and_genmask(
        mesh_path, mask_path, mesh_resolution=64, mask_resolution=16
    )

    # take the union
    binary_voxels = np.logical_or(binary_voxels, ss_mask)
    binary_voxels = torch.from_numpy(binary_voxels).float().cuda()[None, None]
    # the summation of `ss_mask2` is about 44
    ss_mask2 = torch.from_numpy(ss_mask2).float().cuda()[None, None]

    with torch.no_grad():
        encoded = encoder(binary_voxels, sample_posterior=False)

    image = Image.open(image_path).convert("RGB")

    # Run experiments with different parameter combinations
    run_experiments(
        encoded, slat, ss_mask2, image, mask_object, center_and_scale, savedir
    )
