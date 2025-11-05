"""
Headless Batch Rendering with RGB Depth-Based Segmentation
Usage: blender -b your_file.blend -P render_batches.py

Red (closest 5-20), Green (middle 21-35), Blue (farthest 36-50)
"""
import bpy
import os
import random
import numpy as np
import shutil
import gc
import subprocess
import time

# ============================================================================
# CONFIG
# ============================================================================
total_images = 2500
batch_size = 50

successive_window_chance = 0.69
chance_of_occlusion = 0.20

x_bound = 5
y_bound = 5
z_bound = 50

window_names = ["Window.001", "Window.002", "Window.003"]
object_names = ["Cone", "Cube", "Torus", "Cylinder"]

# Depth ranges for RGB segmentation
depth_ranges = [
    (-20, -5),   # Window.001 - Closest (Red channel)
    (-35, -21),  # Window.002 - Middle (Green channel)
    (-50, -36)   # Window.003 - Farthest (Blue channel)
]

# ============================================================================
# SETUP
# ============================================================================
blend_dir = bpy.path.abspath("//")
print(f"\n{'='*70}")
print(f"🎬 BLENDER HEADLESS BATCH RENDERER - RGB DEPTH SEGMENTATION")
print(f"{'='*70}")
print(f".blend file directory: {blend_dir}")

if blend_dir == "":
    raise Exception("⚠️ Please save your .blend file before running this script")

images_folder = os.path.join(blend_dir, "images", "images")
segmented_folder = os.path.join(blend_dir, "images", "segmented")

print(f"RGB output:  {images_folder}")
print(f"Mask output: {segmented_folder}")

# ============================================================================
# GPU CONFIG
# ============================================================================
print(f"\n{'='*70}")
print("⚙️  GPU CONFIGURATION")
print(f"{'='*70}")

bpy.context.scene.render.engine = 'CYCLES'
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
bpy.context.scene.cycles.device = 'GPU'
bpy.context.scene.cycles.samples = 8
bpy.context.scene.render.film_transparent = True

print("✓ Render engine: CYCLES")
print("✓ Device: GPU (CUDA)")
print("✓ Samples: 8")
print("✓ Transparent background: ON")

# ============================================================================
# FOLDER SETUP
# ============================================================================
print(f"\n{'='*70}")
print("📁 FOLDER SETUP")
print(f"{'='*70}")

def clean_folder(folder_path):
    """Delete and recreate folder"""
    if os.path.exists(folder_path):
        try:
            files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
            file_count = len(files)
        except:
            file_count = 0
        shutil.rmtree(folder_path)
        print(f"✓ Deleted {folder_path} ({file_count} files)")
    os.makedirs(folder_path, exist_ok=True)
    print(f"✓ Created {folder_path}")

clean_folder(images_folder)
clean_folder(segmented_folder)

# ============================================================================
# CONFIGURE RGB SEGMENTATION COMPOSITOR
# ============================================================================
print(f"\n{'='*70}")
print("🎨 CONFIGURING RGB SEGMENTATION COMPOSITOR")
print(f"{'='*70}")

# Assign pass indices to windows
for idx, window_name in enumerate(window_names):
    window = bpy.data.objects.get(window_name)
    if window:
        window.pass_index = idx + 1
        print(f"✓ {window_name}: pass_index = {idx + 1}")
    else:
        print(f"⚠️  Warning: {window_name} not found in scene")

# Enable Object Index pass
bpy.context.view_layer.use_pass_object_index = True
print("✓ Enabled Object Index Pass")

# Setup compositor
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
tree.nodes.clear()

# Create nodes
render_layers = tree.nodes.new('CompositorNodeRLayers')
render_layers.location = (0, 400)

# RGB Output setup
rgb_output = tree.nodes.new('CompositorNodeOutputFile')
rgb_output.name = "RGB Output"
rgb_output.label = "RGB Output"
rgb_output.location = (600, 400)
rgb_output.base_path = images_folder
rgb_output.format.file_format = 'PNG'
rgb_output.file_slots.clear()
rgb_output.file_slots.new('rgb')

# Create ID Mask nodes for each window
id_masks = []
for i in range(3):
    id_mask = tree.nodes.new('CompositorNodeIDMask')
    id_mask.index = i + 1
    id_mask.location = (300, 200 - i * 150)
    id_mask.name = f"Window {i+1} Mask"
    id_masks.append(id_mask)
    print(f"✓ Created ID Mask node for Window.{i+1:03d} (index {i+1})")

# Combine RGB node
combine_rgba = tree.nodes.new('CompositorNodeCombineColor')
combine_rgba.mode = 'RGB'
combine_rgba.location = (600, 0)
combine_rgba.label = "RGB Mask Combiner"

# Mask output
mask_output = tree.nodes.new('CompositorNodeOutputFile')
mask_output.name = "Mask Output"
mask_output.label = "Mask Output"
mask_output.location = (800, 0)
mask_output.base_path = segmented_folder
mask_output.format.file_format = 'PNG'
mask_output.format.color_mode = 'RGB'
mask_output.file_slots.clear()
mask_output.file_slots.new('mask')

# Connect nodes
tree.links.new(render_layers.outputs['Image'], rgb_output.inputs['rgb'])
tree.links.new(render_layers.outputs['IndexOB'], id_masks[0].inputs['ID value'])
tree.links.new(render_layers.outputs['IndexOB'], id_masks[1].inputs['ID value'])
tree.links.new(render_layers.outputs['IndexOB'], id_masks[2].inputs['ID value'])
tree.links.new(id_masks[0].outputs['Alpha'], combine_rgba.inputs['Red'])
tree.links.new(id_masks[1].outputs['Alpha'], combine_rgba.inputs['Green'])
tree.links.new(id_masks[2].outputs['Alpha'], combine_rgba.inputs['Blue'])
tree.links.new(combine_rgba.outputs['Image'], mask_output.inputs['mask'])

print("✓ Compositor configured successfully!")
print("  - RGB channels: Red=Window.001, Green=Window.002, Blue=Window.003")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_gpu_usage():
    """Print GPU memory usage via nvidia-smi"""
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits"
        ])
        used, total = map(int, out.decode().strip().split(","))
        print(f"    [GPU] {used}/{total} MB used ({100*used//total}%)")
    except:
        pass

def randomize_scene():
    """Randomize window, object, and light positions"""
    # Randomize windows with depth-based regions
    for window_idx, window_name in enumerate(window_names):
        window = bpy.data.objects.get(window_name)
        if not window:
            continue
            
        if random.random() >= successive_window_chance**(window_idx):
            window.location = (0, 0, 1000)  # Hide off-screen
        else:
            # Use depth range specific to this window
            depth = random.uniform(depth_ranges[window_idx][0], depth_ranges[window_idx][1])
            window.location = (
                random.uniform(-x_bound, x_bound) * abs(depth) / 10,
                random.uniform(-y_bound, y_bound) * abs(depth) / 10,
                depth
            )
            window.rotation_euler = (
                random.uniform(-np.pi * 0.15, np.pi * 0.15),
                random.uniform(-np.pi * 0.15, np.pi * 0.15),
                random.uniform(-np.pi * 0.2, np.pi * 0.2)
            )
    
    # Randomize occlusion objects
    for obj_name in object_names:
        obj = bpy.data.objects.get(obj_name)
        if not obj:
            continue
            
        if random.random() >= chance_of_occlusion:
            obj.location = (0, 0, 1000)
        else:
            depth = random.uniform(-z_bound, 2)
            obj.location = (
                random.uniform(-x_bound, x_bound) * depth / 10,
                random.uniform(-y_bound, y_bound) * depth / 10,
                depth
            )
            obj.rotation_euler = (
                random.uniform(-np.pi, np.pi),
                random.uniform(-np.pi, np.pi),
                random.uniform(-np.pi, np.pi)
            )
    
    # Randomize lighting
    light = bpy.data.objects.get("Light")
    if light:
        light.location = (
            random.uniform(-10, 10),
            random.uniform(-10, 10),
            random.uniform(5, 15)
        )
        light.data.use_shadow = True
        light.data.energy = random.uniform(1000, 10000)

# ============================================================================
# RENDERING
# ============================================================================

def render_batches():
    """Render all images in manageable batches"""
    scene = bpy.context.scene
    
    print(f"\n{'='*70}")
    print(f"🎬 RENDERING {total_images} IMAGES IN BATCHES OF {batch_size}")
    print(f"{'='*70}")
    print(f"Window depth ranges:")
    print(f"  Window.001 (RED):   5-20 units (closest)")
    print(f"  Window.002 (GREEN): 21-35 units (middle)")
    print(f"  Window.003 (BLUE):  36-50 units (farthest)")
    print(f"\nWindow spawn rates (successive_window_chance={successive_window_chance}):")
    print(f"  Window.001: {successive_window_chance**0 * 100:.1f}%")
    print(f"  Window.002: {successive_window_chance**1 * 100:.1f}%")
    print(f"  Window.003: {successive_window_chance**2 * 100:.1f}%")
    print()
    
    total_batches = (total_images + batch_size - 1) // batch_size
    
    for batch_num, batch_start in enumerate(range(1, total_images + 1, batch_size), 1):
        batch_end = min(batch_start + batch_size - 1, total_images)
        
        print(f"\n{'─'*70}")
        print(f"🚀 BATCH {batch_num}/{total_batches}: Rendering images {batch_start}-{batch_end}")
        print(f"{'─'*70}")
        
        batch_start_time = time.time()
        
        for img_idx in range(batch_start, batch_end + 1):
            randomize_scene()
            scene.frame_set(img_idx)
            bpy.ops.render.render(write_still=False)
            
            # Progress indicator
            if img_idx % 50 == 0:
                elapsed = time.time() - batch_start_time
                imgs_done = img_idx - batch_start + 1
                rate = imgs_done / elapsed if elapsed > 0 else 0
                print(f"  [{img_idx}/{batch_end}] {rate:.1f} img/sec", end="")
                print_gpu_usage()
            
            # Periodic garbage collection
            if img_idx % 100 == 0:
                gc.collect()
        
        batch_elapsed = time.time() - batch_start_time
        batch_rate = (batch_end - batch_start + 1) / batch_elapsed
        print(f"\n✅ Batch {batch_num} complete! ({batch_elapsed:.1f}s, {batch_rate:.2f} img/sec)")
        
        gc.collect()
        time.sleep(0.3)

# ============================================================================
# FILE RENAMING
# ============================================================================

def rename_files(folder, prefix):
    """Rename from 'prefix0001.png' to 'prefix_00000.png'"""
    if not os.path.exists(folder):
        return 0
    
    files = []
    for f in os.listdir(folder):
        if f.startswith(prefix) and f.endswith('.png'):
            files.append(f)
    
    files.sort()
    
    renamed = 0
    for i, old_filename in enumerate(files):
        old_path = os.path.join(folder, old_filename)
        new_filename = f"{prefix}_{i:05d}.png"
        new_path = os.path.join(folder, new_filename)
        
        if old_path != new_path:
            os.rename(old_path, new_path)
            renamed += 1
    
    return renamed

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    start_time = time.time()
    
    print_gpu_usage()
    render_batches()
    
    # Rename files
    print(f"\n{'='*70}")
    print("📝 RENAMING FILES")
    print(f"{'='*70}")
    
    rgb_renamed = rename_files(images_folder, "rgb")
    print(f"✓ Renamed {rgb_renamed} RGB files")
    
    mask_renamed = rename_files(segmented_folder, "mask")
    print(f"✓ Renamed {mask_renamed} mask files")
    
    # Final verification
    rgb_files = len([f for f in os.listdir(images_folder) if f.endswith('.png')])
    mask_files = len([f for f in os.listdir(segmented_folder) if f.endswith('.png')])
    
    total_elapsed = time.time() - start_time
    
    print(f"\n{'='*70}")
    print("✅✅✅ RENDERING COMPLETE! ✅✅✅")
    print(f"{'='*70}")
    print(f"\nResults:")
    print(f"  RGB images:  {rgb_files}")
    print(f"  Mask images: {mask_files}")
    print(f"  Total time:  {total_elapsed/60:.1f} minutes ({total_elapsed:.1f}s)")
    print(f"  Average:     {total_images/total_elapsed:.2f} images/sec")
    print(f"\nLocation:")
    print(f"  {images_folder}")
    print(f"  {segmented_folder}")
    
    if rgb_files == total_images and mask_files == total_images:
        print("\n✓ Perfect! All files generated successfully")
        print("\n📊 RGB Mask Encoding (Depth-Based):")
        print("    RED channel   = Window.001 (5-20 units, CLOSEST)")
        print("    GREEN channel = Window.002 (21-35 units, middle)")
        print("    BLUE channel  = Window.003 (36-50 units, FARTHEST)")
        print("\n    Pixel values:")
        print("      [255, 0, 0]   = Red window only (closest)")
        print("      [0, 255, 0]   = Green window only")
        print("      [0, 0, 255]   = Blue window only (farthest)")
        print("      [255, 255, 0] = Red + Green overlap")
        print("      [0, 0, 0]     = Background (no windows)")
    else:
        print(f"\n⚠️  Warning: Expected {total_images} of each type!")
    
    print(f"\n{'='*70}\n")