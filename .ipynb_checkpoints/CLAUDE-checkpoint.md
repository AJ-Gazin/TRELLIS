# TRELLIS Local Editing Project - Claude Documentation

## Project Overview
This TRELLIS fork implements experimental local editing capabilities for 3D meshes. The primary goal is to test and demonstrate the TRELLIS method for mesh editing, allowing local geometry changes to objects while maintaining overall structure.

**Date**: July 2025 (Updated)
**Status**: ✅ **WORKING** - Complete tested setup with one-command installation
**Repository**: https://github.com/AJ-Gazin/TRELLIS (based on https://github.com/FishWoWater/TRELLIS)

## 🚀 Quick Start (< 10 minutes)

### One-Command Setup
```bash
git clone https://github.com/AJ-Gazin/TRELLIS.git
cd TRELLIS
./quick_setup.sh
```

### Run Local Editing
```bash
conda activate trellis
export ATTN_BACKEND=flash-attn
export SPCONV_ALGO=native
python example_local_edit.py
```

**That's it!** Results will be in `results/local_edit/`

## Core Objective
Test a TRELLIS-based method for mesh editing. The authors claimed in their paper that local editing of objects is possible, but never published the code. This fork implements it as an experimental feature with a fully tested working environment.

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
- NVIDIA GPU with at least 16GB memory (tested on NVIDIA A40 with 46GB VRAM)
- Linux system (tested on Ubuntu 22.04)

### Software
- **Python**: 3.10+ (installed by quick_setup.sh)
- **CUDA**: 12.6/12.7 compatible drivers
- **PyTorch**: 2.6.0 with cu126 support (installed automatically)

## ✅ Tested Working Configuration (July 2025)

### Core Dependencies
- **torch==2.6.0+cu126** (PyTorch with CUDA 12.6)
- **spconv-cu126==2.3.8** (Sparse convolution)
- **flash-attn==2.8.0.post2** (Attention backend)
- **nvdiffrast==0.3.3** (NVIDIA differentiable rasterization)
- **diff-gaussian-rasterization** (Gaussian splatting)
- **simple-knn** (K-nearest neighbors for point clouds)
- **diffoctreerast** (Octree-based differentiable rendering)

### Mesh Processing
- **pymeshlab==2023.12.post3** (Mesh processing)
- **pymeshfix==0.17.1** (Mesh repair)
- **pyvista==0.45.2** (3D data processing)
- **trimesh==4.6.12** (Mesh loading/processing)
- **open3d==0.19.0** (3D data processing)
- **utils3d==0.0.2** (3D utilities from specific commit)

### Environment Variables
```bash
export ATTN_BACKEND=flash-attn    # Use flash-attn instead of xformers
export SPCONV_ALGO=native         # Recommended for single runs
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

## Output Structure
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

Models are automatically downloaded by `quick_setup.sh`.

## Key Fixes Applied in This Fork

### 1. kaolin Compatibility Issue ✅
- **Problem**: Original code tried to import `kaolin.utils.testing.check_tensor` but kaolin has compatibility issues with PyTorch 2.6.0
- **Solution**: Replaced kaolin import with complete `check_tensor` implementation in `trellis/representations/mesh/flexicubes/flexicubes.py` including `throw` parameter support
- **Impact**: FlexiCubes mesh extraction now works without requiring kaolin

### 2. spconv Version Compatibility ✅
- **Problem**: Original setup used spconv-cu120 which had CUDA library conflicts with CUDA 12.7
- **Solution**: Updated to spconv-cu126 (version 2.3.8) which is compatible with CUDA 12.6/12.7
- **Impact**: Sparse convolution backend now works properly

### 3. PyTorch Version Selection ✅
- **Problem**: Latest PyTorch 2.7+ had compatibility issues with various dependencies
- **Solution**: Selected PyTorch 2.6.0 + cu126 as the stable, tested combination
- **Impact**: All components work together without version conflicts

### 4. Attention Backend Selection ✅
- **Problem**: xformers had compatibility issues with our PyTorch version
- **Solution**: Used flash-attn as the attention backend instead
- **Impact**: Attention operations work correctly with better performance

### 5. Complete Rendering Dependencies ✅
- **Problem**: Missing diff-gaussian-rasterization, simple-knn, diffoctreerast
- **Solution**: Added all rendering dependencies to setup script
- **Impact**: Full pipeline works including Gaussian splatting, radiance fields, and mesh rendering

## Manual Installation (if needed)

If you need to customize the installation, here's the manual process:

```bash
# Create conda environment
conda create -n trellis python=3.10
conda activate trellis

# Install PyTorch 2.6.0 with CUDA 12.6 support
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# Install core dependencies
pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime trimesh open3d transformers igraph xatlas

# Install mesh processing
pip install pymeshlab pymeshfix pyvista plyfile

# Install attention and sparse conv
pip install flash-attn spconv-cu126

# Install rendering dependencies
pip install git+https://github.com/NVlabs/nvdiffrast.git
pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git@main
pip install git+https://github.com/camenduru/simple-knn.git@main

# Install diffoctreerast
git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/diffoctreerast
pip install /tmp/diffoctreerast
rm -rf /tmp/diffoctreerast

# Install utils3d
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# Initialize submodules
git submodule update --init --recursive

# Set environment variables
export ATTN_BACKEND=flash-attn
export SPCONV_ALGO=native
```

## Verification Commands

```bash
# Test environment setup
conda activate trellis
export ATTN_BACKEND=flash-attn
export SPCONV_ALGO=native

# Test core imports
python -c "from trellis import models; from trellis.pipelines import TrellisImageTo3DPipeline; print('✅ TRELLIS imports successful')"

# Test mesh loading
python -c "import trimesh; scene = trimesh.load('./assets/example_local_edit/doll2/doll.glb'); print(f'✅ Mesh loaded: {len(list(scene.geometry.values())[0].vertices)} vertices')"

# Test GPU and CUDA
python -c "import torch; print(f'✅ PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Test rendering dependencies
python -c "import diff_gaussian_rasterization; import simple_knn; import diffoctreerast; print('✅ All rendering dependencies')"
```

## Troubleshooting

### Environment Issues
- Always activate conda environment: `conda activate trellis`
- Set environment variables: `export ATTN_BACKEND=flash-attn && export SPCONV_ALGO=native`
- If imports fail, run verification commands above

### Memory Issues
- Requires 16GB+ GPU memory
- Use `low_vram=True` in pipeline initialization if needed
- Close other GPU applications

### Model Download Issues
- Models auto-download on first run
- Check internet connection
- Verify HuggingFace hub access

## Environment Persistence

Add to your shell profile (`~/.bashrc` or `~/.zshrc`):
```bash
# TRELLIS environment variables
export ATTN_BACKEND=flash-attn
export SPCONV_ALGO=native

# Quick TRELLIS activation
alias trellis="cd /path/to/TRELLIS && conda activate trellis && export ATTN_BACKEND=flash-attn && export SPCONV_ALGO=native"
```

## Code References
- `example_local_edit.py:138` - Main encoder initialization
- `example_local_edit.py:212` - Core editing pipeline call
- `example_local_edit.py:147` - Example configuration list
- `trellis/representations/mesh/flexicubes/flexicubes.py:13` - Fixed check_tensor implementation

## Documentation & References

### TRELLIS Paper
- **File**: `TRELLIS_paper.pdf` (38MB)
- **Source**: https://arxiv.org/pdf/2412.01506
- **Title**: "Structured 3D Latents for Scalable and Versatile 3D Generation"
- **Section 3.4**: Image-conditioned local editing methodology

This paper is essential reading for understanding the local editing algorithms and theoretical foundation.

---

**Last Updated**: July 2025  
**Status**: ✅ Working setup verified with successful local editing runs  
**Setup Time**: < 10 minutes with `quick_setup.sh`