# Creating 3D Masks for TRELLIS Local Editing

## Understanding 3D Masks

### How 3D Masks Differ from 2D Masks
- **2D Masks**: Binary images where white = edit, black = keep
- **3D Masks**: 3D geometry that defines a volume in space
- **Key Difference**: 3D masks have depth and must spatially overlap the region you want to edit

### What Makes a Good 3D Mask
1. **Slightly larger** than the target region (10-20% bigger)
2. **Watertight** geometry (closed mesh)
3. **Simple geometry** (boxes/spheres work well)
4. **Same coordinate system** as original model

## Method 1: Automated Script (Easiest)

For tables and furniture with predictable structure:

```python
import trimesh
import numpy as np

def create_table_leg_mask(table_glb_path, output_path):
    # Load table
    scene = trimesh.load(table_glb_path)
    mesh = list(scene.geometry.values())[0] if isinstance(scene, trimesh.Scene) else scene
    
    # Get dimensions
    bounds = mesh.bounds
    table_height = bounds[1][2] - bounds[0][2]
    
    # Create box covering bottom 40% of table
    mask = trimesh.creation.box(
        extents=[
            bounds[1][0] - bounds[0][0] + 0.1,  # Width + margin
            bounds[1][1] - bounds[0][1] + 0.1,  # Depth + margin  
            table_height * 0.4                   # Bottom 40%
        ]
    )
    
    # Position at table bottom
    mask.apply_translation([
        (bounds[0][0] + bounds[1][0]) / 2,   # Center X
        (bounds[0][1] + bounds[1][1]) / 2,   # Center Y
        bounds[0][2] + table_height * 0.2    # Bottom-aligned
    ])
    
    # Export
    mask.export(output_path)
```

## Method 2: Blender (Visual Method)

### Step-by-Step Instructions:

1. **Import Your Model**
   - Open Blender
   - File → Import → glTF 2.0 (.glb/.gltf)
   - Select your table.glb

2. **Create Mask Geometry**
   - Press `Shift+A` → Mesh → Cube
   - Press `S` to scale, `G` to move
   - Position cube(s) to cover table legs

3. **Adjust Mask Size**
   - Press `Tab` for Edit mode
   - Select all with `A`
   - Press `S` then `1.1` to scale up 10%
   - Press `Tab` to exit Edit mode

4. **Delete Original Table**
   - Select the table mesh
   - Press `X` → Delete

5. **Export Mask**
   - File → Export → glTF 2.0
   - Name it `legs_mask.glb`
   - Settings:
     - Format: glTF Binary (.glb)
     - Include: Selected Objects only

## Method 3: MeshLab (Free Alternative)

1. **Load Model**
   - File → Import Mesh → select table.glb

2. **Create Geometric Primitives**
   - Filters → Create New Mesh Layer → Geometric Primitives
   - Choose Box or Cylinder
   - Adjust size and position

3. **Duplicate and Position**
   - For multiple legs, duplicate primitives
   - Use Transform tools to position

4. **Merge and Export**
   - Select all mask layers
   - Filters → Mesh Layer → Flatten Visible Layers
   - File → Export as → .glb

## Method 4: Online Tools

### Three.js Editor (Browser-based)
1. Go to: https://threejs.org/editor/
2. File → Import → your table.glb
3. Add → Box geometry
4. Scale and position to cover legs
5. Delete original table
6. File → Export GLB

### Tips for All Methods:

1. **Check Alignment**
   - Import both table.glb and mask.glb back into any 3D viewer
   - Ensure mask covers desired region

2. **Common Mask Patterns for Furniture**
   - **Table legs**: Box covering bottom 40-50% of height
   - **Chair legs**: 4 small boxes at corners
   - **Lamp base**: Cylinder at bottom
   - **Cabinet doors**: Flat boxes on front face

3. **Testing Your Mask**
   - Start with larger masks (better coverage)
   - Reduce size if edits affect too much
   - Use simple shapes (boxes/cylinders)

## Python Script for Common Cases

Save this as `create_furniture_mask.py`:

```python
import trimesh
import click

@click.command()
@click.option('--input', '-i', required=True, help='Input GLB file')
@click.option('--output', '-o', required=True, help='Output mask GLB')
@click.option('--type', '-t', type=click.Choice(['legs', 'top', 'front', 'custom']), default='legs')
@click.option('--height-ratio', '-h', default=0.4, help='Height ratio for legs (0-1)')
def create_mask(input, output, type, height_ratio):
    """Create masks for furniture editing"""
    
    scene = trimesh.load(input)
    mesh = list(scene.geometry.values())[0] if isinstance(scene, trimesh.Scene) else scene
    bounds = mesh.bounds
    
    if type == 'legs':
        # Box covering bottom portion
        size = bounds[1] - bounds[0]
        mask = trimesh.creation.box([
            size[0] * 1.1,
            size[1] * 1.1,
            size[2] * height_ratio
        ])
        mask.apply_translation([
            (bounds[0][0] + bounds[1][0]) / 2,
            (bounds[0][1] + bounds[1][1]) / 2,
            bounds[0][2] + size[2] * height_ratio / 2
        ])
    
    elif type == 'top':
        # Box covering top surface
        size = bounds[1] - bounds[0]
        mask = trimesh.creation.box([
            size[0] * 0.9,
            size[1] * 0.9,
            size[2] * 0.1
        ])
        mask.apply_translation([
            (bounds[0][0] + bounds[1][0]) / 2,
            (bounds[0][1] + bounds[1][1]) / 2,
            bounds[1][2] - size[2] * 0.05
        ])
    
    # Export
    scene_out = trimesh.Scene()
    scene_out.add_geometry(mask)
    scene_out.export(output)
    print(f"Mask saved to {output}")

if __name__ == '__main__':
    create_mask()
```

Usage:
```bash
python create_furniture_mask.py -i table.glb -o legs_mask.glb -t legs -h 0.4
```

## Quick Tips

1. **No 3D experience?** Use the automated script
2. **Visual control needed?** Use Three.js editor (no installation)
3. **Complex masks?** Use Blender (free, powerful)
4. **Batch processing?** Modify the Python script

The key is that the mask just needs to roughly cover the region you want to edit - it doesn't need to be perfect!