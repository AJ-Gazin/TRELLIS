#!/usr/bin/env python
"""
Automated mask creation for furniture local editing in TRELLIS
"""

import trimesh
import click
import numpy as np
import os

@click.command()
@click.option('--input', '-i', required=True, help='Input GLB file')
@click.option('--output', '-o', required=True, help='Output mask GLB')
@click.option('--type', '-t', type=click.Choice(['legs', 'top', 'front', 'custom', 'legs-individual']), default='legs')
@click.option('--height-ratio', '-h', default=0.4, help='Height ratio for legs (0-1)')
@click.option('--margin', '-m', default=0.1, help='Margin to add around mask (0-1)')
def create_mask(input, output, type, height_ratio, margin):
    """Create masks for furniture editing"""
    
    # Load the model
    scene = trimesh.load(input)
    if isinstance(scene, trimesh.Scene):
        # Get the first geometry
        mesh = list(scene.geometry.values())[0]
    else:
        mesh = scene
    
    # Get bounding box
    bounds = mesh.bounds
    size = bounds[1] - bounds[0]
    center = (bounds[0] + bounds[1]) / 2
    
    print(f"Model bounds: {bounds}")
    print(f"Model size: {size}")
    
    if type == 'legs':
        # Single box covering entire bottom portion
        mask = trimesh.creation.box([
            size[0] * (1 + margin),
            size[1] * (1 + margin),
            size[2] * height_ratio
        ])
        mask.apply_translation([
            center[0],
            center[1],
            bounds[0][2] + size[2] * height_ratio / 2
        ])
        
    elif type == 'legs-individual':
        # Create individual boxes for each corner (typical 4-leg table)
        leg_size = min(size[0], size[1]) * 0.15  # Assume legs are ~15% of table width
        masks = []
        
        # Define corner offsets
        corners = [
            (-1, -1),  # Front-left
            (1, -1),   # Front-right
            (-1, 1),   # Back-left
            (1, 1),    # Back-right
        ]
        
        for x_sign, y_sign in corners:
            box = trimesh.creation.box([
                leg_size * (1 + margin),
                leg_size * (1 + margin),
                size[2] * height_ratio
            ])
            
            # Position at corner
            box.apply_translation([
                center[0] + (size[0]/2 - leg_size/2) * x_sign * 0.8,
                center[1] + (size[1]/2 - leg_size/2) * y_sign * 0.8,
                bounds[0][2] + size[2] * height_ratio / 2
            ])
            
            masks.append(box)
        
        # Combine all leg masks
        mask = trimesh.util.concatenate(masks)
        
    elif type == 'top':
        # Box covering top surface
        mask = trimesh.creation.box([
            size[0] * (1 - margin),
            size[1] * (1 - margin),
            size[2] * 0.1
        ])
        mask.apply_translation([
            center[0],
            center[1],
            bounds[1][2] - size[2] * 0.05
        ])
        
    elif type == 'front':
        # Box covering front face
        mask = trimesh.creation.box([
            size[0] * (1 + margin),
            size[1] * 0.1,
            size[2] * (1 + margin)
        ])
        mask.apply_translation([
            center[0],
            bounds[0][1] + size[1] * 0.05,
            center[2]
        ])
    
    # Create scene and export
    scene_out = trimesh.Scene()
    scene_out.add_geometry(mask)
    scene_out.export(output)
    
    print(f"✅ Mask saved to {output}")
    print(f"Mask type: {type}")
    if type in ['legs', 'legs-individual']:
        print(f"Height ratio: {height_ratio} ({height_ratio * 100}% of model height)")
    print(f"Margin: {margin * 100}%")

@click.command()
@click.option('--input', '-i', required=True, help='Input GLB file')
def analyze_model(input):
    """Analyze a 3D model to help with mask creation"""
    
    scene = trimesh.load(input)
    if isinstance(scene, trimesh.Scene):
        mesh = list(scene.geometry.values())[0]
        print(f"Scene with {len(scene.geometry)} objects")
    else:
        mesh = scene
    
    bounds = mesh.bounds
    size = bounds[1] - bounds[0]
    
    print(f"\n📊 Model Analysis: {os.path.basename(input)}")
    print(f"{'='*50}")
    print(f"Vertices: {len(mesh.vertices)}")
    print(f"Faces: {len(mesh.faces)}")
    print(f"\nBounding Box:")
    print(f"  Min: [{bounds[0][0]:.3f}, {bounds[0][1]:.3f}, {bounds[0][2]:.3f}]")
    print(f"  Max: [{bounds[1][0]:.3f}, {bounds[1][1]:.3f}, {bounds[1][2]:.3f}]")
    print(f"\nDimensions:")
    print(f"  Width (X): {size[0]:.3f}")
    print(f"  Depth (Y): {size[1]:.3f}")
    print(f"  Height (Z): {size[2]:.3f}")
    print(f"\nSuggested leg heights:")
    print(f"  30%: {size[2] * 0.3:.3f}")
    print(f"  40%: {size[2] * 0.4:.3f}")
    print(f"  50%: {size[2] * 0.5:.3f}")

@click.group()
def cli():
    """TRELLIS Furniture Mask Creation Tool"""
    pass

cli.add_command(create_mask, name='create')
cli.add_command(analyze_model, name='analyze')

if __name__ == '__main__':
    cli()