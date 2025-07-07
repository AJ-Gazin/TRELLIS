# TRELLIS Local Editing: Step-by-Step Explanation

## Overview
Local editing in TRELLIS allows you to modify specific regions of a 3D model while preserving the rest. This document explains how the existing doll example works, step by step.

## Example Structure: `assets/example_local_edit/`

```
assets/example_local_edit/
├── doll/                     # First example (not used by default)
│   ├── doll.glb             # Original 3D model
│   ├── head_lantern.png     # Reference image for editing
│   ├── head_mask.glb        # 3D mask defining edit region
│   └── latents/
│       ├── slat.npz         # Pre-computed high-res features
│       └── ss.npz           # Pre-computed sparse structure
└── doll2/                   # Main example (used in script)
    ├── doll.glb             # Original 3D model
    ├── head_furry.png       # Reference: furry head style
    ├── head_lantern.png     # Reference: lantern head style
    ├── head_mailbox.png     # Reference: mailbox head style
    ├── head_mask.glb        # Mask: entire head region
    ├── headcenter_mask.glb  # Mask: center of head only
    └── latents/
        ├── slat.npz         # Pre-computed features
        └── ss.npz           # Pre-computed structure
```

## How Local Editing Works: Step-by-Step

### Step 1: Load Pre-computed Latents
```python
ss, slat = load_ss_and_slat(latent_dir)
```
- **File Used**: `doll2/latents/ss.npz`, `doll2/latents/slat.npz`
- **Purpose**: These contain the 3D model encoded into TRELLIS's latent space
- **SS (Sparse Structure)**: 16³ voxel grid encoding overall shape
- **SLat (Structured Latent)**: 64³ feature grid with detailed information

### Step 2: Load and Process the Mask
```python
mask_object = create_o3d_mesh(bake_scene_to_mesh(mask_path))
```
- **File Used**: `doll2/head_mask.glb` or `doll2/headcenter_mask.glb`
- **Purpose**: Defines which part of the model to edit
- **Process**: 
  - Loads the GLB file as a 3D mesh
  - Converts to Open3D format for voxelization

### Step 3: Voxelize Original Model and Mask
```python
center_and_scale, binary_voxels, ss_mask, ss_mask2 = voxelize_and_genmask(
    mesh_path, mask_path, mesh_resolution=64, mask_resolution=16
)
```
- **Files Used**: `doll2/doll.glb` (original), mask file
- **Purpose**: Convert 3D meshes to voxel grids
- **Outputs**:
  - `binary_voxels`: 64³ voxelized original model
  - `ss_mask`: 64³ voxelized mask
  - `ss_mask2`: 16³ low-res mask (matches SS resolution)
  - `center_and_scale`: Normalization parameters

### Step 4: Encode Voxels to Latent Space
```python
with torch.no_grad():
    encoded = encoder(binary_voxels, sample_posterior=False)
```
- **Input**: Combined voxels (original OR mask)
- **Output**: Encoded sparse structure representation
- **Purpose**: Convert voxel representation to TRELLIS latent

### Step 5: Load Reference Image
```python
image = Image.open(image_path).convert("RGB")
```
- **Files Used**: `doll2/head_lantern.png`, etc.
- **Purpose**: Provides visual guidance for how to edit the masked region
- **Format**: RGB image, typically 512×512 or 1024×1024

### Step 6: Run Local Editing Pipeline
```python
outputs_all = pipeline.run_local_editing(
    encoded,           # Encoded voxels
    slat,             # Pre-computed detailed features
    ss_mask2,         # 16³ mask
    image,            # Reference image
    mask_object=mask_object,  # Original mask mesh
    ...
)
```
- **Process**:
  1. Uses diffusion to modify the sparse structure in masked regions
  2. Applies image conditioning to guide the generation
  3. Resamples multiple times for refinement
  4. Generates multiple output formats

### Step 7: Save Results
The pipeline generates:
- **Videos**: `sample_gs.mp4` (Gaussian), `sample_rf.mp4` (Radiance Field), `sample_mesh.mp4`
- **3D Model**: `edited.glb` - Final edited mesh with textures
- **Latents**: Updated `ss.npz` and `slat.npz` for the edited model

## Key Concepts

### 1. **3D Masks vs 2D Masks**
- **2D Masks**: Define pixels to edit in an image (binary: edit/don't edit)
- **3D Masks**: Define volumes in 3D space to edit
- Unlike 2D, 3D masks must account for depth and internal structure
- The mask is a separate 3D model that spatially overlaps the edit region

### 2. **Multi-Resolution Processing**
- **64³ resolution**: For detailed mesh processing
- **16³ resolution**: For sparse structure editing
- The mask is voxelized at both resolutions to match different processing stages

### 3. **Latent Preservation**
- Pre-computed latents preserve the original model's structure
- Only the masked region is modified during diffusion
- This ensures non-masked areas remain unchanged

### 4. **Image Conditioning**
- Reference images guide the style/appearance of edits
- The model learns to map image features to 3D geometry
- Multiple reference images can create different variations

## Example Configurations in Script

The script tests three scenarios:
1. **Full head replacement** (`head_mask.glb` + `head_lantern.png`)
2. **Center head edit** (`headcenter_mask.glb` + `head_lantern.png`)
3. **Different styles** (same mask with `head_furry.png`, `head_mailbox.png`)

Each combination demonstrates different editing possibilities:
- Mask size affects edit scope
- Reference images control final appearance
- Same mask + different images = style variations

## Parameter Effects

- **Steps** (12, 25): More steps = higher quality but slower
- **CFG Strength** (5.0-7.5): Higher = stronger image adherence
- **Resample Times** (1-5): More resampling = better integration
- **Resolution**: Affects detail level and memory usage