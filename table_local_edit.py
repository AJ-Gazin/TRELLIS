#!/usr/bin/env python
# Table leg editing experiment

import os
os.environ["ATTN_BACKEND"] = "flash-attn"
os.environ["SPCONV_ALGO"] = "native"

import sys
sys.path.append(os.path.dirname(__file__))

from example_local_edit import *

# Configuration for table experiments
table_experiments = [
    (
        "table_legs_metal",
        "./assets/example_local_edit/table/latents/",
        "./assets/example_local_edit/table/table.glb",
        "./assets/example_local_edit/table/legs_mask.glb",
        "./assets/example_local_edit/table/legs_metal.png",
    ),
    (
        "table_legs_ornate",
        "./assets/example_local_edit/table/latents/",
        "./assets/example_local_edit/table/table.glb",
        "./assets/example_local_edit/table/legs_mask.glb",
        "./assets/example_local_edit/table/legs_ornate.png",
    ),
    (
        "table_legs_modern",
        "./assets/example_local_edit/table/latents/",
        "./assets/example_local_edit/table/table.glb",
        "./assets/example_local_edit/table/legs_mask.glb",
        "./assets/example_local_edit/table/legs_modern.png",
    ),
]

# Run table experiments
for example in table_experiments:
    instance_name, latent_dir, mesh_path, mask_path, image_path = example
    savedir = os.path.join("results/local_edit", instance_name)
    os.makedirs(savedir, exist_ok=True)
    
    print(f"\nProcessing {instance_name}...")
    
    # Check if files exist
    if not os.path.exists(mesh_path):
        print(f"ERROR: Table model not found at {mesh_path}")
        continue
    if not os.path.exists(mask_path):
        print(f"ERROR: Mask model not found at {mask_path}")
        continue
    if not os.path.exists(image_path):
        print(f"ERROR: Reference image not found at {image_path}")
        continue
    if not os.path.exists(os.path.join(latent_dir, "ss.npz")):
        print(f"ERROR: Latents not found in {latent_dir}")
        print("Please run generate_table_latents.py first")
        continue
    
    # Load latents
    ss, slat = load_ss_and_slat(latent_dir)
    mask_object = create_o3d_mesh(bake_scene_to_mesh(mask_path))
    
    # Voxelize and generate masks
    center_and_scale, binary_voxels, ss_mask, ss_mask2 = voxelize_and_genmask(
        mesh_path, mask_path, mesh_resolution=64, mask_resolution=16
    )
    
    # Process voxels
    binary_voxels = np.logical_or(binary_voxels, ss_mask)
    binary_voxels = torch.from_numpy(binary_voxels).float().cuda()[None, None]
    ss_mask2 = torch.from_numpy(ss_mask2).float().cuda()[None, None]
    
    # Encode
    with torch.no_grad():
        encoded = encoder(binary_voxels, sample_posterior=False)
    
    # Load reference image
    image = Image.open(image_path).convert("RGB")
    
    # Run with optimized parameters for furniture
    # Using fewer variations for initial testing
    steps = [20]  # Slightly more steps for furniture detail
    cfg_strengths = [(7.5, 5.0)]  # Moderate strength for controlled edits
    resample_times_list = [2]  # 2 resamples usually sufficient
    
    for step in steps:
        for cfg_strength in cfg_strengths:
            cfg_ss, cfg_slat = cfg_strength
            for resample_times in resample_times_list:
                experiment_dir = os.path.join(
                    savedir,
                    f"cfg_{cfg_ss:.1f}_{cfg_slat:.1f}_resample{resample_times}_steps{step}",
                )
                
                if os.path.exists(os.path.join(experiment_dir, "edited.glb")):
                    print(f"Skipping existing: {experiment_dir}")
                    continue
                
                print(f"Running: steps={step}, cfg=({cfg_ss}, {cfg_slat}), resample={resample_times}")
                
                try:
                    outputs_all = pipeline.run_local_editing(
                        encoded,
                        slat,
                        ss_mask2,
                        image,
                        seed=42,  # Fixed seed for reproducibility
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
                        slat_sampler_params={
                            "steps": step,
                            "cfg_strength": cfg_slat,
                        },
                        enable_slat_repaint=False,
                        return_all=True,
                    )
                    
                    # Save outputs
                    outputs = outputs_all["outputs"]
                    os.makedirs(experiment_dir, exist_ok=True)
                    os.makedirs(os.path.join(experiment_dir, "latents"), exist_ok=True)
                    
                    # Save latents
                    np.savez(
                        os.path.join(experiment_dir, "latents", "ss.npz"),
                        mean=outputs_all["ss_coords"],
                    )
                    np.savez(
                        os.path.join(experiment_dir, "latents", "slat.npz"),
                        coords=outputs_all["slat_coords"],
                        feats=outputs_all["slat_feats"],
                    )
                    
                    # Render videos
                    video = render_utils.render_video(outputs["gaussian"][0])["color"]
                    imageio.mimsave(os.path.join(experiment_dir, "sample_gs.mp4"), video, fps=30)
                    
                    video = render_utils.render_video(outputs["radiance_field"][0])["color"]
                    imageio.mimsave(os.path.join(experiment_dir, "sample_rf.mp4"), video, fps=30)
                    
                    video = render_utils.render_video(outputs["mesh"][0])["normal"]
                    imageio.mimsave(os.path.join(experiment_dir, "sample_mesh.mp4"), video, fps=30)
                    
                    # Export GLB
                    glb = postprocessing_utils.to_trimesh(
                        outputs["radiance_field"][0],
                        outputs["mesh"][0],
                        texture_bake_mode="fast",
                        postprocess_mode="simplify",
                        simplify=0.95,
                        texture_size=1024,
                    )
                    glb.export(os.path.join(experiment_dir, "edited.glb"))
                    
                    print(f"✅ Saved to {experiment_dir}")
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()

print("\n✨ Table experiments complete! Check results/local_edit/")