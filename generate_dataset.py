"""
Blender Dataset Generation Script for GapFlyt Project
Place in Code/ directory and run with: 
  cd Code/
  blender -b ../Blender/Window.blend --python generate_dataset.py
"""

import bpy
import os
from pathlib import Path

# Paths relative to Code/ directory
CODE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = CODE_DIR.parent
BLENDER_DIR = PROJECT_ROOT / "Blender"
ASSETS_DIR = BLENDER_DIR / "Assets"
OUTPUT_DIR = BLENDER_DIR / "Outputs"
IMAGES_DIR = OUTPUT_DIR / "Images"
MASKS_DIR = OUTPUT_DIR / "GTMasks"

# Create output directories
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
MASKS_DIR.mkdir(parents=True, exist_ok=True)

# Asset paths
TEXTURES_DIR = ASSETS_DIR / "Textures"
WINDOWS_DIR = ASSETS_DIR / "Windows"

# Get texture and mask files
TEXTURES = sorted(list(TEXTURES_DIR.glob("*.jpg")))[:4]
WINDOW_MASKS = sorted(list(WINDOWS_DIR.glob("*.png")))[:4]

print("=" * 70)
print("GapFlyt Dataset Generation Script")
print("=" * 70)
print(f"Script location: {CODE_DIR}")
print(f"Project root: {PROJECT_ROOT}")
print(f"Blender dir: {BLENDER_DIR}")
print(f"Output dir: {OUTPUT_DIR}")
print(f"\nTextures found: {len(TEXTURES)}")
for t in TEXTURES:
    print(f"  - {t.name}")
print(f"\nWindow masks found: {len(WINDOW_MASKS)}")
for w in WINDOW_MASKS:
    print(f"  - {w.name}")
print(f"\nTotal combinations: {len(TEXTURES) * len(WINDOW_MASKS) * 2} (2 camera angles)")
print("=" * 70)

# Scene setup
scene = bpy.context.scene
scene.render.image_settings.file_format = 'PNG'
scene.render.resolution_percentage = 100

print("\n=== Scene Information ===")
print(f"Mesh objects: {[obj.name for obj in bpy.data.objects if obj.type == 'MESH']}")
print(f"Cameras: {[obj.name for obj in bpy.data.objects if obj.type == 'CAMERA']}")
print(f"Resolution: {scene.render.resolution_x}x{scene.render.resolution_y}")
print(f"Engine: {scene.render.engine}")

# Find the wall/plane object
wall_objects = [obj for obj in bpy.data.objects 
                if obj.type == 'MESH' and ('Wall' in obj.name or 'Plane' in obj.name)]

if not wall_objects:
    wall_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']

if not wall_objects:
    print("\nERROR: No mesh objects found in scene!")
    import sys
    sys.exit(1)

wall = wall_objects[0]
print(f"\nUsing wall object: {wall.name}")

# Find cameras (for 2 different viewpoints)
cameras = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']
if len(cameras) >= 2:
    CAMERAS = cameras[:2]
    print(f"Using cameras: {[cam.name for cam in CAMERAS]}")
elif len(cameras) == 1:
    # Use same camera for both angles (you can modify camera position in loop if needed)
    CAMERAS = [cameras[0], cameras[0]]
    print(f"Using single camera: {cameras[0].name} for both angles")
else:
    print("ERROR: No cameras found!")
    import sys
    sys.exit(1)

def apply_texture_and_mask(wall_obj, texture_path, mask_path):
    """Apply texture with alpha mask to wall object"""
    
    # Get or create material
    if wall_obj.data.materials:
        mat = wall_obj.data.materials[0]
    else:
        mat = bpy.data.materials.new(name="WallMaterial")
        wall_obj.data.materials.append(mat)
    
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    # Output node
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (600, 0)
    
    # BSDF
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (300, 0)
    
    # Texture image
    tex_image = nodes.new('ShaderNodeTexImage')
    tex_image.location = (-300, 200)
    tex_image.image = bpy.data.images.load(str(texture_path), check_existing=True)
    
    # Mask image  
    mask_image = nodes.new('ShaderNodeTexImage')
    mask_image.location = (-300, -200)
    mask_image.image = bpy.data.images.load(str(mask_path), check_existing=True)
    
    # Connect texture to base color
    links.new(tex_image.outputs['Color'], bsdf.inputs['Base Color'])
    
    # Connect mask to alpha (use color as alpha value)
    links.new(mask_image.outputs['Color'], bsdf.inputs['Alpha'])
    
    # Connect BSDF to output
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    # Enable transparency
    mat.blend_method = 'BLEND'
    mat.shadow_method = 'NONE'
    
    return mat

def setup_mask_material(wall_obj, mask_path):
    """Setup pure mask material for GT mask rendering"""
    
    if wall_obj.data.materials:
        mat = wall_obj.data.materials[0]
    else:
        mat = bpy.data.materials.new(name="MaskMaterial")
        wall_obj.data.materials.append(mat)
    
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (300, 0)
    
    emission = nodes.new('ShaderNodeEmission')
    emission.location = (0, 0)
    emission.inputs['Strength'].default_value = 1.0
    
    mask_image = nodes.new('ShaderNodeTexImage')
    mask_image.location = (-300, 0)
    mask_image.image = bpy.data.images.load(str(mask_path), check_existing=True)
    
    links.new(mask_image.outputs['Color'], emission.inputs['Color'])
    links.new(emission.outputs['Emission'], output.inputs['Surface'])
    
    return mat

# Main generation loop
combination_id = 0
total = len(TEXTURES) * len(WINDOW_MASKS) * len(CAMERAS)

print(f"\n{'=' * 70}")
print("Starting Generation")
print(f"{'=' * 70}\n")

for cam_idx, camera in enumerate(CAMERAS):
    scene.camera = camera
    cam_name = f"Cam{cam_idx + 1}"
    
    for tex_idx, tex_path in enumerate(TEXTURES):
        tex_name = tex_path.stem
        
        for mask_idx, mask_path in enumerate(WINDOW_MASKS):
            mask_name = mask_path.stem
            
            combination_id += 1
            
            print(f"[{combination_id:02d}/{total}] {cam_name} | {tex_name} | {mask_name}")
            
            # Filenames
            img_file = f"img_{combination_id:03d}_c{cam_idx}_t{tex_idx}_m{mask_idx}.png"
            mask_file = f"mask_{combination_id:03d}_c{cam_idx}_t{tex_idx}_m{mask_idx}.png"
            
            # Render RGB image
            apply_texture_and_mask(wall, tex_path, mask_path)
            scene.render.image_settings.color_mode = 'RGBA'
            scene.render.filepath = str(IMAGES_DIR / img_file)
            bpy.ops.render.render(write_still=True)
            print(f"  ✓ RGB: {img_file}")
            
            # Render GT mask
            setup_mask_material(wall, mask_path)
            scene.render.image_settings.color_mode = 'BW'
            scene.render.filepath = str(MASKS_DIR / mask_file)
            bpy.ops.render.render(write_still=True)
            print(f"  ✓ Mask: {mask_file}")

print(f"\n{'=' * 70}")
print("Generation Complete!")
print(f"{'=' * 70}")
print(f"Generated {combination_id} image pairs")
print(f"RGB images: {IMAGES_DIR}")
print(f"GT masks: {MASKS_DIR}")
print(f"{'=' * 70}")