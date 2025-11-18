"""
Blender Dataset Generation Script for GapFlyt Project - Parallax Sequences
Generates multi-view sequences of the same scene for TS2P learning

Each sequence contains:
- Multiple camera viewpoints of the same scene configuration
- Single ground truth mask for that sequence
"""

import bpy
import os
from pathlib import Path
import math
import mathutils

# Paths relative to Code/ directory
CODE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = CODE_DIR.parent
BLENDER_DIR = PROJECT_ROOT / "Blender"
ASSETS_DIR = BLENDER_DIR / "Assets"
OUTPUT_DIR = BLENDER_DIR / "Outputs"
SEQUENCES_DIR = OUTPUT_DIR / "Sequences"

# Clean up and create output directory
if SEQUENCES_DIR.exists():
    import shutil
    shutil.rmtree(SEQUENCES_DIR)
SEQUENCES_DIR.mkdir(parents=True, exist_ok=True)

# Asset paths
TEXTURES_DIR = ASSETS_DIR / "Textures"
WINDOWS_DIR = ASSETS_DIR / "Windows"

# Get texture and mask files
TEXTURES = sorted(list(TEXTURES_DIR.glob("*.jpg")))[:4]
WINDOW_MASKS = sorted(list(WINDOWS_DIR.glob("*.png")))[:4]

# Augmentation parameters
ROTATIONS = [0, 15, 30, 45]  # Window rotation angles in degrees
SCALES = [0.9, 1.0, 1.1]     # Window scale factors (90%, 100%, 110%)

# TS2P Learning Parameters
# Generate MORE frames than deployment uses → enables random subset sampling during training
CAMERA_POSITIONS = 10  # Total frames generated (1 ref + 9 scan positions)
DEPLOYMENT_SUBSET = 5  # What model uses at test time (1 ref + 4 scan)
SCAN_DISTANCE = 2.0    # Total diagonal movement distance in meters (increased for larger scene)
CAMERA_START_DISTANCE = 11.0  # Initial distance from window (matching your scene)

print("=" * 70)
print("GapFlyt TS2P Learning Dataset - Diagonal Scanning")
print("=" * 70)
print(f"Strategy: Generate {CAMERA_POSITIONS} frames, train on random {DEPLOYMENT_SUBSET}-frame subsets")
print(f"Why? → Model learns TS2P logic, not just one scanning pattern")
print("=" * 70)
print(f"Script location: {CODE_DIR}")
print(f"Output dir: {OUTPUT_DIR}")
print(f"\nTextures: {len(TEXTURES)}")
for t in TEXTURES:
    print(f"  - {t.name}")
print(f"\nWindow masks: {len(WINDOW_MASKS)}")
for w in WINDOW_MASKS:
    print(f"  - {w.name}")
print(f"\nAugmentations:")
print(f"  Rotations: {ROTATIONS}")
print(f"  Scales: {SCALES}")
print(f"\nCamera scanning:")
print(f"  Frames per sequence: {CAMERA_POSITIONS} (1 reference + {CAMERA_POSITIONS-1} scan)")
print(f"  Scan type: Diagonal (forward + sideways)")
print(f"  Scan distance: {SCAN_DISTANCE}m")
print(f"  Start distance: {CAMERA_START_DISTANCE}m")

total_sequences = len(TEXTURES) * len(WINDOW_MASKS) * len(ROTATIONS) * len(SCALES)
total_images = total_sequences * CAMERA_POSITIONS
print(f"\nTotal sequences: {total_sequences}")
print(f"Total images: {total_images}")
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

# Find the Window object (foreground wall) - NOT the Background
window_obj = bpy.data.objects.get('Window')

if not window_obj:
    window_objects = [obj for obj in bpy.data.objects 
                     if obj.type == 'MESH' and 'Window' in obj.name]
    if window_objects:
        window_obj = window_objects[0]
    else:
        print("\nERROR: Window object not found!")
        import sys
        sys.exit(1)

wall = window_obj
print(f"\nUsing wall object: {wall.name}")
print(f"  Location: {wall.location}")
print(f"  Scale: {wall.scale}")
print(f"  Dimensions: {wall.dimensions}")

# Calculate appropriate scale
current_size = max(wall.dimensions.x, wall.dimensions.y)
recommended_size = CAMERA_START_DISTANCE * 0.6  # Window should be ~60% of distance
if current_size > CAMERA_START_DISTANCE * 0.8:
    print(f"⚠ WARNING: Window is very large ({current_size:.1f}m)")
    print(f"  Recommended size: {recommended_size:.1f}m")
    print(f"  You may want to reduce window scale in Blender")
    print(f"  Or the generation script will apply scale factors (0.9-1.1)")

# Ensure wall has UV mapping (critical for textures!)
if not wall.data.uv_layers:
    print("⚠ No UV mapping found! Adding UV coordinates...")
    bpy.context.view_layer.objects.active = wall
    wall.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.unwrap(method='ANGLE_BASED')
    bpy.ops.object.mode_set(mode='OBJECT')
    wall.select_set(False)
    print("✓ UV mapping added!")
else:
    print(f"✓ UV mapping exists ({len(wall.data.uv_layers)} layers)")

# CRITICAL: Check if window normals are facing the right direction
# Window should face +Z so camera behind it (+Z) can see it
mesh = wall.data
if len(mesh.polygons) > 0:
    face_normal = mesh.polygons[0].normal
    print(f"Window face normal: ({face_normal.x:.2f}, {face_normal.y:.2f}, {face_normal.z:.2f})")
    
    if face_normal.z < 0:
        print("⚠ WARNING: Window normals face backward! Flipping...")
        # Select the object and flip normals
        bpy.context.view_layer.objects.active = wall
        wall.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.flip_normals()
        bpy.ops.object.mode_set(mode='OBJECT')
        wall.select_set(False)
        print("✓ Normals flipped!")
    else:
        print("✓ Window normals face forward")

# Get or create camera
cameras = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']
if cameras:
    camera = cameras[0]
else:
    print("ERROR: No camera found!")
    import sys
    sys.exit(1)

scene.camera = camera
print(f"Using camera: {camera.name}")

# Store original camera location
original_camera_location = camera.location.copy()
original_camera_rotation = camera.rotation_euler.copy()

def apply_texture_and_mask(wall_obj, texture_path, mask_path, rotation=0, scale=1.0):
    """Apply texture with alpha mask to wall object with rotation and scale"""
    
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
    output.location = (900, 0)
    
    # BSDF
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (600, 0)
    
    # Texture coordinate node
    tex_coord = nodes.new('ShaderNodeTexCoord')
    tex_coord.location = (-1200, 0)
    
    # Mapping node for rotation and scale
    mapping = nodes.new('ShaderNodeMapping')
    mapping.location = (-900, 0)
    mapping.inputs['Rotation'].default_value[2] = math.radians(rotation)
    mapping.inputs['Scale'].default_value = (scale, scale, scale)
    
    # Texture image
    tex_image = nodes.new('ShaderNodeTexImage')
    tex_image.location = (-600, 200)
    tex_image.image = bpy.data.images.load(str(texture_path), check_existing=True)
    
    # Mask image  
    mask_image = nodes.new('ShaderNodeTexImage')
    mask_image.location = (-600, -200)
    mask_image.image = bpy.data.images.load(str(mask_path), check_existing=True)
    
    # Connect nodes
    links.new(tex_coord.outputs['UV'], mapping.inputs['Vector'])
    links.new(mapping.outputs['Vector'], tex_image.inputs['Vector'])
    links.new(mapping.outputs['Vector'], mask_image.inputs['Vector'])
    links.new(tex_image.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(mask_image.outputs['Color'], bsdf.inputs['Alpha'])
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    mat.blend_method = 'BLEND'
    mat.shadow_method = 'NONE'
    
    return mat

def setup_mask_material(wall_obj, mask_path, rotation=0, scale=1.0):
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
    output.location = (600, 0)
    
    emission = nodes.new('ShaderNodeEmission')
    emission.location = (300, 0)
    emission.inputs['Strength'].default_value = 1.0
    
    tex_coord = nodes.new('ShaderNodeTexCoord')
    tex_coord.location = (-600, 0)
    
    mapping = nodes.new('ShaderNodeMapping')
    mapping.location = (-300, 0)
    mapping.inputs['Rotation'].default_value[2] = math.radians(rotation)
    mapping.inputs['Scale'].default_value = (scale, scale, scale)
    
    mask_image = nodes.new('ShaderNodeTexImage')
    mask_image.location = (0, 0)
    mask_image.image = bpy.data.images.load(str(mask_path), check_existing=True)
    
    links.new(tex_coord.outputs['UV'], mapping.inputs['Vector'])
    links.new(mapping.outputs['Vector'], mask_image.inputs['Vector'])
    links.new(mask_image.outputs['Color'], emission.inputs['Color'])
    links.new(emission.outputs['Emission'], output.inputs['Surface'])
    
    return mat

def set_camera_diagonal_scan(camera, scan_step, total_steps, start_distance, look_at=(0, 0, 0)):
    """
    Position camera along diagonal scanning trajectory (GapFlyt style)
    Uses Y-forward coordinate system (Y is depth axis)
    
    Args:
        scan_step: Current step in scan (0 = reference frame)
        total_steps: Total number of steps
        start_distance: Starting distance from window along Y axis
        look_at: Point camera looks at
    """
    # Get window Z height from scene to match
    window_z = look_at[2] if isinstance(look_at, (list, tuple)) else 0
    
    if scan_step == 0:
        # Reference frame position - camera behind window looking at it
        # Y-negative is "behind" in this coordinate system
        camera.location.x = 0
        camera.location.y = -start_distance  # Negative Y = behind window
        camera.location.z = window_z  # Match window height
    else:
        # Diagonal scanning motion in XY plane
        progress = scan_step / (total_steps - 1)  # 0 to 1
        
        # Move diagonally: sideways (X) and forward toward window (increasing Y from negative)
        camera.location.x = -progress * SCAN_DISTANCE / math.sqrt(2)  # Move left
        camera.location.y = -start_distance + progress * SCAN_DISTANCE / math.sqrt(2)  # Move forward (toward 0)
        camera.location.z = window_z  # Keep at window height
    
    # Point camera at window (look in +Y direction)
    direction = mathutils.Vector(look_at) - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')  # -Z is camera forward
    camera.rotation_euler = rot_quat.to_euler()

# Main generation loop
sequence_id = 0

print(f"\n{'=' * 70}")
print("Starting Sequence Generation")
print(f"{'=' * 70}\n")

for tex_idx, tex_path in enumerate(TEXTURES):
    tex_name = tex_path.stem
    
    for mask_idx, mask_path in enumerate(WINDOW_MASKS):
        mask_name = mask_path.stem
        
        for rot_idx, rotation in enumerate(ROTATIONS):
            
            for scale_idx, scale in enumerate(SCALES):
                
                sequence_id += 1
                sequence_name = f"seq_{sequence_id:04d}_t{tex_idx}_m{mask_idx}_r{rot_idx}_s{scale_idx}"
                sequence_dir = SEQUENCES_DIR / sequence_name
                sequence_dir.mkdir(exist_ok=True)
                
                print(f"[{sequence_id:03d}/{total_sequences}] {sequence_name}")
                print(f"  Texture: {tex_name} | Mask: {mask_name} | Rot:{rotation}° | Scale:{scale:.1f}")
                
                # Setup material once for this sequence
                apply_texture_and_mask(wall, tex_path, mask_path, rotation=rotation, scale=scale)
                
                # Generate frames along diagonal scanning trajectory
                # Frame 0 = reference frame
                # Frames 1 to N = scanning trajectory  
                for frame_idx in range(CAMERA_POSITIONS):
                    # Position camera for this frame
                    set_camera_diagonal_scan(
                        camera, 
                        scan_step=frame_idx,
                        total_steps=CAMERA_POSITIONS,
                        start_distance=CAMERA_START_DISTANCE,
                        look_at=(wall.location.x, wall.location.y, wall.location.z)
                    )
                    
                    # Generate RGB frame
                    scene.render.image_settings.color_mode = 'RGBA'
                    frame_file = sequence_dir / f"frame_{frame_idx:02d}.png"
                    scene.render.filepath = str(frame_file)
                    bpy.ops.render.render(write_still=True)
                    
                    # Generate corresponding mask from same camera position
                    setup_mask_material(wall, mask_path, rotation=rotation, scale=scale)
                    scene.render.image_settings.color_mode = 'BW'
                    mask_file = sequence_dir / f"mask_{frame_idx:02d}.png"
                    scene.render.filepath = str(mask_file)
                    bpy.ops.render.render(write_still=True)
                    
                    # Switch back to RGB material for next frame
                    apply_texture_and_mask(wall, tex_path, mask_path, rotation=rotation, scale=scale)
                
                print(f"  ✓ Generated {CAMERA_POSITIONS} frames + {CAMERA_POSITIONS} masks")
                
                # Save sequence metadata
                import json
                metadata = {
                    "sequence_id": sequence_id,
                    "sequence_name": sequence_name,
                    "texture": tex_name,
                    "mask": mask_name,
                    "rotation_deg": rotation,
                    "scale": scale,
                    "num_frames": CAMERA_POSITIONS,
                    "scan_distance": SCAN_DISTANCE,
                    "scan_type": "diagonal",
                    "camera_start_distance": CAMERA_START_DISTANCE,
                    "frame_naming": "frame_XX.png = RGB frame, mask_XX.png = corresponding GT mask",
                    "note": "Each frame has its own ground truth mask from the same camera position"
                }
                
                with open(sequence_dir / "metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)

# Reset camera to original position
camera.location = original_camera_location
camera.rotation_euler = original_camera_rotation

print(f"\n{'=' * 70}")
print("Generation Complete!")
print(f"{'=' * 70}")
print(f"Generated {sequence_id} sequences")
print(f"Total images: {sequence_id * CAMERA_POSITIONS}")
print(f"Output: {SEQUENCES_DIR}")
print("=" * 70)