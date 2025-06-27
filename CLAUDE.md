# TRELLIS Local Editing Project - Claude Documentation

## Project Overview
This TRELLIS fork implements experimental local editing capabilities for 3D meshes. The primary goal is to test and demonstrate the TRELLIS method for mesh editing, allowing local geometry changes to objects while maintaining overall structure.

**Date**: December 2024
**Status**: Working implementation with Claude Code tested setup
**Repository**: https://github.com/AJ-Gazin/TRELLIS (based on https://github.com/FishWoWater/TRELLIS)

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
- **Python**: 3.10+ (tested with 3.10.18)
- **CUDA**: 12.7 (or 12.6+)
- **PyTorch**: 2.6.0 with cu126 support (tested working combination)

## Quick Setup (Claude Code Working Configuration)

### One-Command Install
```bash
# Create conda environment
conda create -n trellis python=3.10
conda activate trellis

# Install using Claude Code tested configuration
./setup.sh --claude-working-setup

# Set environment variables
export ATTN_BACKEND=flash-attn
export SPCONV_ALGO=native
```

### Manual Installation (if you need to customize)

#### Environment Setup
```bash
# Create conda environment
conda create -n trellis python=3.10
conda activate trellis

# Install PyTorch 2.6.0 with CUDA 12.6 support (tested working)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# Install basic dependencies
pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime trimesh open3d transformers

# Install utils3d from specific commit (required for TRELLIS)
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# Install spconv-cu126 (latest version compatible with CUDA 12.6)
pip install spconv-cu126

# Install flash-attn (works better than xformers with our setup)
pip install flash-attn

# Install mesh processing dependencies
pip install pymeshlab pymeshfix pyvista

# Install nvdiffrast from source
pip install git+https://github.com/NVlabs/nvdiffrast.git

# Install additional dependencies
pip install plyfile
```

### Environment Variables
```bash
export ATTN_BACKEND=flash-attn    # Use flash-attn instead of xformers
export SPCONV_ALGO=native         # Recommended for single runs
```

## Tested Package Versions (Working Configuration)
- torch==2.6.0+cu126
- torchvision==0.21.0+cu126
- torchaudio==2.6.0+cu126
- spconv-cu126==2.3.8
- flash-attn==2.8.0.post2
- nvdiffrast==0.3.3
- pymeshlab==2023.12.post3
- pymeshfix==0.17.1
- pyvista==0.45.2
- trimesh==4.6.12
- open3d==0.19.0
- utils3d==0.0.2

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

## Key Fixes Applied in This Fork

### 1. kaolin Compatibility Issue
- **Problem**: Original code tried to import `kaolin.utils.testing.check_tensor` but kaolin 0.17.0 has compatibility issues with PyTorch 2.6.0
- **Solution**: Replaced kaolin import with minimal `check_tensor` implementation in `trellis/representations/mesh/flexicubes/flexicubes.py`
- **Impact**: FlexiCubes mesh extraction now works without requiring kaolin

### 2. spconv Version Compatibility  
- **Problem**: Original setup used spconv-cu120 which had CUDA library conflicts with CUDA 12.7
- **Solution**: Updated to spconv-cu126 (version 2.3.8) which is compatible with CUDA 12.6/12.7
- **Impact**: Sparse convolution backend now works properly

### 3. PyTorch Version Selection
- **Problem**: Latest PyTorch 2.7+ had compatibility issues with various dependencies
- **Solution**: Selected PyTorch 2.6.0 + cu126 as the stable, tested combination
- **Impact**: All components work together without version conflicts

### 4. Attention Backend Selection
- **Problem**: xformers had compatibility issues with our PyTorch version
- **Solution**: Used flash-attn as the attention backend instead
- **Impact**: Attention operations work correctly with better performance

## Verification Commands

### Test Environment Setup
```bash
# Activate environment
conda activate trellis
export ATTN_BACKEND=flash-attn
export SPCONV_ALGO=native

# Test core imports
python -c "from trellis import models; from trellis.pipelines import TrellisImageTo3DPipeline; print('✅ TRELLIS imports successful')"

# Test mesh loading
python -c "import trimesh; mesh = trimesh.load('./assets/example_local_edit/doll2/doll.glb'); print(f'✅ Mesh loaded: {len(mesh.vertices)} vertices')"

# Test GPU and CUDA
python -c "import torch; print(f'✅ PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

## Known Limitations

- Experimental implementation - results may vary
- Requires substantial GPU memory (16GB+) 
- Limited to objects that fit TRELLIS training distribution
- Mask quality significantly affects editing results
- kaolin functionality is limited (only basic check_tensor implemented)

## Resuming from this Checkpoint

This repository is now set up with a working TRELLIS local editing environment. To resume work:

### 1. Clone the Fork
```bash
git clone https://github.com/AJ-Gazin/TRELLIS.git
cd TRELLIS
git checkout dev  # Make sure you're on the dev branch with our changes
```

### 2. Quick Setup
```bash
# Create and activate conda environment
conda create -n trellis python=3.10
conda activate trellis

# Install using our tested configuration
./setup.sh --claude-working-setup

# Set environment variables for the session
export ATTN_BACKEND=flash-attn
export SPCONV_ALGO=native
```

### 3. Verify Setup
```bash
# Test the environment
python -c "from trellis import models; from trellis.pipelines import TrellisImageTo3DPipeline; print('🎉 TRELLIS ready for local editing!')"
```

### 4. Next Steps
- Source table mesh for furniture demo (`assets/example_local_edit/table/`)
- Create editing masks and reference images
- Run `example_local_edit.py` experiments
- Generate visualization results for client demo

### Environment Persistence
Add to your shell profile (`~/.bashrc` or `~/.zshrc`):
```bash
# TRELLIS environment variables
export ATTN_BACKEND=flash-attn
export SPCONV_ALGO=native
```

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