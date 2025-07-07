#!/usr/bin/env python
# Script to generate presaved latents for table model from existing mesh

import os
os.environ["ATTN_BACKEND"] = "flash-attn"  
os.environ["SPCONV_ALGO"] = "native"

import numpy as np
import open3d as o3d
import torch
import trimesh
import logging
from datetime import datetime
from trellis import models
from trellis.pipelines import TrellisImageTo3DPipeline

# Setup logging
os.makedirs("logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"logs/generate_table_latents_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"Starting table latent generation - log file: {log_file}")

def bake_scene_to_mesh(mesh_path: str):
    """Load and process GLB mesh"""
    mesh: trimesh.Scene = trimesh.load(mesh_path)
    geo_name = list(mesh.geometry.keys())[0]
    geo = mesh.geometry[geo_name]
    nodes = mesh.graph.geometry_nodes[geo_name]
    transform = mesh.graph[nodes[0]][0]
    geo.apply_transform(transform)
    return geo

def create_o3d_mesh(mesh: trimesh.Trimesh):
    """Convert trimesh to Open3D mesh"""
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices),
        o3d.utility.Vector3iVector(mesh.faces),
    )
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    return mesh

def voxelize(mesh: o3d.geometry.TriangleMesh, resolution: int = 64):
    """Voxelize mesh at given resolution"""
    # Clamp vertices to [-0.5, 0.5] range
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

def normalize_mesh(mesh: trimesh.Trimesh):
    """Normalize mesh to [-0.5, 0.5] bounds"""
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    scale = (bounds[1] - bounds[0]).max()
    
    mesh.vertices = (mesh.vertices - center) / (scale + 1e-8)
    return mesh, center, scale

# Setup paths
table_mesh_path = "assets/table/table_1.glb"
output_dir = "assets/table/latents"
os.makedirs(output_dir, exist_ok=True)

logger.info("Loading table mesh...")
try:
    table_mesh = bake_scene_to_mesh(table_mesh_path)
    table_mesh, center, scale = normalize_mesh(table_mesh)
    
    logger.info(f"Table dimensions: {table_mesh.bounds[1] - table_mesh.bounds[0]}")
    logger.info(f"Normalization - center: {center}, scale: {scale}")
    
    # Convert to Open3D and voxelize
    o3d_mesh = create_o3d_mesh(table_mesh)
    binary_voxels = voxelize(o3d_mesh, resolution=64)
    
    logger.info(f"Voxelized with {binary_voxels.sum()} occupied voxels")
    
except Exception as e:
    logger.error(f"Error loading/processing table mesh: {e}")
    raise

# Load encoder
logger.info("Loading encoder...")
try:
    enc_pretrained = "microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16"
    encoder = models.from_pretrained(enc_pretrained).eval().cuda()
    logger.info("Encoder loaded successfully")
    
    # Convert to tensor and encode
    binary_voxels_tensor = torch.from_numpy(binary_voxels).float().cuda()[None, None]
    
    logger.info("Encoding to sparse structure latent...")
    with torch.no_grad():
        encoded = encoder(binary_voxels_tensor, sample_posterior=False)
    
    # Save sparse structure latent
    ss_output_path = os.path.join(output_dir, "ss.npz")
    np.savez(ss_output_path, mean=encoded.cpu().numpy())
    logger.info(f"Sparse structure latent saved to {ss_output_path}")
    
except Exception as e:
    logger.error(f"Error during sparse structure encoding: {e}")
    raise

# Generate slat using pipeline
logger.info("Generating slat using pipeline...")
try:
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()
    logger.info("Pipeline loaded successfully")
    
    # Create a simple reference image for slat generation (we'll use the reference image)
    from PIL import Image
    ref_image = Image.open("assets/table/table_leg_reference.png").convert("RGB")
    logger.info("Reference image loaded")
    
    # Generate structured latent using the proper pipeline flow
    logger.info("Running pipeline to generate slat...")
    
    # Hook into the sample_slat method to capture the slat
    captured_slat = [None]
    original_sample_slat = pipeline.sample_slat
    
    def capture_sample_slat(*args, **kwargs):
        result = original_sample_slat(*args, **kwargs)
        captured_slat[0] = result
        logger.info(f"Captured slat with coords shape: {result.coords.shape}, feats shape: {result.feats.shape}")
        return result
    
    # Temporarily replace the sample_slat method
    pipeline.sample_slat = capture_sample_slat
    
    try:
        outputs = pipeline.run(
            ref_image,
            seed=42,
            sparse_structure_sampler_params={
                "steps": 12,
                "cfg_strength": 7.5,
            },
            slat_sampler_params={
                "steps": 12,
                "cfg_strength": 3,
            },
        )
    finally:
        # Restore original sample_slat method
        pipeline.sample_slat = original_sample_slat
    
    # Extract and save slat
    slat_saved = False
    slat_output_path = os.path.join(output_dir, "slat.npz")
    
    # Method 1: Use captured slat
    if captured_slat[0] is not None:
        np.savez(
            slat_output_path,
            coords=captured_slat[0].coords.cpu().numpy(),
            feats=captured_slat[0].feats.cpu().numpy()
        )
        logger.info(f"Slat saved from captured state to {slat_output_path}")
        slat_saved = True
    
    # Method 2: Check if latents are in outputs 
    elif "latents" in outputs and "slat" in outputs["latents"]:
        slat = outputs["latents"]["slat"]
        np.savez(
            slat_output_path,
            coords=slat.coords.cpu().numpy(),
            feats=slat.feats.cpu().numpy()
        )
        logger.info(f"Slat saved from outputs to {slat_output_path}")
        slat_saved = True
    
    # Method 3: Check pipeline sampler state
    elif hasattr(pipeline.slat_sampler, 'slat') and pipeline.slat_sampler.slat is not None:
        slat = pipeline.slat_sampler.slat
        np.savez(
            slat_output_path,
            coords=slat.coords.cpu().numpy(),
            feats=slat.feats.cpu().numpy()
        )
        logger.info(f"Slat saved from sampler state to {slat_output_path}")
        slat_saved = True
    
    if not slat_saved:
        logger.error("CRITICAL: Could not extract slat from pipeline")
        logger.info(f"Available output keys: {list(outputs.keys())}")
        logger.info("Pipeline attributes:")
        if hasattr(pipeline, 'slat_sampler'):
            logger.info(f"  slat_sampler attributes: {[attr for attr in dir(pipeline.slat_sampler) if not attr.startswith('_')]}")
        logger.error("Cannot proceed with local editing without proper slat!")
        raise RuntimeError("Failed to generate slat - this is required for local editing")
        
except Exception as e:
    logger.error(f"Error during slat generation: {e}")
    raise

logger.info("✅ Latents generation complete!")
logger.info(f"   Sparse structure: {ss_output_path}")
logger.info(f"   Slat: {slat_output_path if 'slat_output_path' in locals() else 'NOT GENERATED'}")
logger.info(f"   Normalization info - center: {center}, scale: {scale}")
logger.info(f"   Ready for local editing with table_leg_reference.png")
logger.info(f"   Log file: {log_file}")