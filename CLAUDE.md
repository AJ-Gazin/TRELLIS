# TRELLIS Local Editing Project - Claude Documentation

## Project Overview
This TRELLIS fork implements experimental local editing capabilities for 3D meshes. The primary goal is to test and demonstrate the TRELLIS method for mesh editing, allowing local geometry changes to objects while maintaining overall structure.

**Date**: June 2025
**Status**: Experimental local editing implementation testing

## Core Objective
Test a TRELLIS-based method for mesh editing. The authors claimed in their paper that local editing of objects is possible, but never published the code. This fork implements it as an experimental feature.

### Primary Testing Pipeline
1. Take a 3D model of furniture (table recommended for visible geometry changes)
2. Generate presaved latents & voxelized mask using TRELLIS pipeline
3. Run `example_local_edit.py` on the model/latents/mask to demonstrate local geometry changes
4. Create visualizations showing:
   - Original mesh
   - Voxelized mask applied
   - Reference image used
   - Final edited results

## System Requirements

### Hardware
- NVIDIA GPU with at least 16GB memory (tested on A100, A6000)
- Linux system (recommended)

### Software
- **Python**: 3.12
- **CUDA**: 12.7
- **PyTorch**: Must use `cu126` variant (torch with CUDA 12.6 support for compatibility)

## Installation

### Environment Setup
```bash
# Create conda environment with Python 3.12
conda create -n trellis python=3.12
conda activate trellis

# Install PyTorch with CUDA 12.6 support (compatible with CUDA 12.7)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install project dependencies
. ./setup.sh --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast

# For local editing specific dependencies
pip install trimesh open3d imageio pillow tqdm numpy
```

### Environment Variables
```bash
export ATTN_BACKEND=xformers  # or flash-attn
export SPCONV_ALGO=native     # Recommended for single runs
```

## Local Editing Pipeline

### Key Script: `example_local_edit.py`
This script implements the experimental local editing feature with the following workflow:

1. **Load Presaved Latents**: Loads sparse structure (ss.npz) and slat (slat.npz) from assets directory
2. **Mesh Processing**: Processes original mesh and mask geometry
3. **Voxelization**: Creates voxelized representations at different resolutions
4. **Local Editing**: Applies image-conditioned editing to specified regions
5. **Output Generation**: Produces edited 3D models in multiple formats

### Example Assets Structure
```
assets/example_local_edit/
├── doll/                     # Example 1
│   ├── doll.glb             # Original mesh
│   ├── head_lantern.png     # Reference image
│   ├── head_mask.glb        # Editing mask
│   └── latents/
│       ├── slat.npz         # Structured latent
│       └── ss.npz           # Sparse structure
└── doll2/                   # Example 2 (multiple variants)
    ├── doll.glb
    ├── head_furry.png
    ├── head_lantern.png
    ├── head_mailbox.png
    ├── head_mask.glb
    ├── headcenter_mask.glb
    └── latents/
        ├── slat.npz
        └── ss.npz
```

### Testing Parameters
The script tests various parameter combinations:
- **Steps**: [12, 25] (sampling steps)
- **CFG Strengths**: [(5.0, 3.0), (7.5, 5.0)] (ss_cfg, slat_cfg)
- **Resample Times**: [1, 2, 3, 5]

## Running Local Editing

### Basic Execution
```bash
python example_local_edit.py
```

### Output Structure
Results are saved to `results/local_edit/` with subdirectories for each parameter combination:
```
results/local_edit/
└── [experiment_name]/
    ├── cfg_7.5_5.0_resample2_steps12/
    │   ├── edited.glb           # Final edited mesh
    │   ├── sample_gs.mp4        # Gaussian render
    │   ├── sample_rf.mp4        # Radiance field render
    │   ├── sample_mesh.mp4      # Mesh render
    │   └── latents/
    │       ├── ss.npz           # Generated sparse structure
    │       └── slat.npz         # Generated slat
    └── [other parameter combinations...]
```

## Key Functions

### Mesh Processing
- `bake_scene_to_mesh()`: Processes GLB files to trimesh objects
- `create_o3d_mesh()`: Converts to Open3D format for voxelization
- `voxelize_and_genmask()`: Creates voxel representations and masks

### Latent Handling
- `load_sparse_structure()`: Loads ss.npz files
- `load_slat()`: Loads slat.npz structured latents
- `load_ss_and_slat()`: Combined loading function

### Editing Pipeline
- `run_experiments()`: Main experimental loop testing different parameters
- `pipeline.run_local_editing()`: Core editing function with resampling options

## Model Requirements

### Pretrained Models
- **TRELLIS-image-large**: Primary model (1.2B parameters)
- **Encoder**: `microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16`

Models are automatically downloaded from Hugging Face on first run.

## Experimental Features

### Resampling Methods
- Method 1: Treats x_t as x_0 in forward diffusion
- Multiple resampling iterations for quality improvement

### Postprocessing Options
- Simplification (triangle reduction)
- Remeshing iterations
- Texture baking modes: "fast" for low VRAM

## Testing Workflow for New Objects

1. **Prepare Assets**:
   - Source mesh (preferably furniture like tables)
   - Create editing mask (GLB format)
   - Find/create reference image for editing

2. **Generate Latents**:
   - Use TRELLIS pipeline to encode mesh to latents
   - Save as ss.npz and slat.npz in assets structure

3. **Configure Test**:
   - Add to `example_pairs` list in script
   - Specify paths to mesh, mask, latents, and reference image

4. **Execute and Analyze**:
   - Run script and review generated outputs
   - Compare different parameter combinations
   - Evaluate quality of local edits

## Known Limitations

- Experimental implementation - results may vary
- Requires substantial GPU memory (16GB+)
- Limited to objects that fit TRELLIS training distribution
- Mask quality significantly affects editing results

## Documentation & References

### TRELLIS Paper
- **File**: `TRELLIS_paper.pdf` (38MB)
- **Source**: https://arxiv.org/pdf/2412.01506
- **Title**: "Structured 3D Latents for Scalable and Versatile 3D Generation"
- **Content**: Complete technical documentation including:
  - Detailed SLAT (Structured Latent) methodology
  - Local editing algorithms and theoretical foundation
  - Training procedures and model architectures
  - Experimental results and performance comparisons
  - Implementation details for all components

This paper is essential reading for understanding:
- Section 3.4: Image-conditioned local editing methodology
- Sparse structure and structured latent representations
- Flow matching and rectified flow transformers
- Postprocessing and output format generation

### Code References
- `example_local_edit.py:138` - Main encoder initialization
- `example_local_edit.py:212` - Core editing pipeline call
- `example_local_edit.py:147` - Example configuration list