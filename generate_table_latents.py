#!/usr/bin/env python
# Script to generate presaved latents for table model

import os
os.environ["ATTN_BACKEND"] = "flash-attn"
os.environ["SPCONV_ALGO"] = "native"

import numpy as np
import torch
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline

# Setup
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.cuda()

# You'll need a good image of your table for initial generation
table_image_path = "assets/example_local_edit/table/table_photo.png"  # You need to provide this
output_dir = "assets/example_local_edit/table/latents"
os.makedirs(output_dir, exist_ok=True)

# Load image
image = Image.open(table_image_path).convert("RGB")

# Generate 3D from image
outputs = pipeline.run(
    image,
    seed=42,
    return_intermediate=True,  # This will return the latents
    sparse_structure_sampler_params={
        "steps": 12,
        "cfg_strength": 7.5,
    },
    slat_sampler_params={
        "steps": 12,
        "cfg_strength": 3,
    },
)

# Extract and save latents
if "latents" in outputs:
    # Save sparse structure
    ss_latent = outputs["latents"]["sparse_structure_latent"]
    np.savez(
        os.path.join(output_dir, "ss.npz"),
        mean=ss_latent.cpu().numpy()
    )
    
    # Save slat
    slat = outputs["latents"]["slat"]
    np.savez(
        os.path.join(output_dir, "slat.npz"),
        coords=slat.coords.cpu().numpy(),
        feats=slat.feats.cpu().numpy()
    )
    
    print(f"Latents saved to {output_dir}")
else:
    print("Error: Latents not returned. Check pipeline configuration.")

# Also save the generated mesh for reference
if "mesh" in outputs:
    glb = outputs["mesh"][0]
    glb.export("assets/example_local_edit/table/generated_table.glb")