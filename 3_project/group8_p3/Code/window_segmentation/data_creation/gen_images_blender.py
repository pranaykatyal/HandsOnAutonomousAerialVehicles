import random
import bpy
import numpy as np

# For NVIDIA GPUs
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
bpy.context.scene.cycles.device = 'GPU'
bpy.context.scene.cycles.samples = 8

# get the nodes from the compositor
TrainingFile = bpy.context.scene.node_tree.nodes.get("Training File Output")
SegmentedFile = bpy.context.scene.node_tree.nodes.get("Segmented File Output")

# relative path to the output images
images_folder = "./images/images/"
segmented_folder = "./images/segmented/"
TrainingFile.base_path = images_folder
SegmentedFile.base_path = segmented_folder

bpy.ops.object.select_all(action='DESELECT')

object_names = ["Cone", "Cube", "Torus"]
window_names = ["Window1", "Window2", "Window3", "Window4"]
objects = []

x_bound = 4.5   # experimentally determined bounds for largest window at z = 0 using blender camera view
y_bound = 3
z_bound = 40    # negative; camera is at z = 10
num_images = 100
successive_window_chance = 0.42
chance_of_occlusion = 0.1

cam = bpy.context.scene.camera
for img_idx in range(num_images):

    # randomize the windows
    for window_idx in range(len(window_names)):
        window = bpy.data.objects.get(window_names[window_idx])

        # move the unused windows out of the camera view
        #   the chance will get exponentially smaller for each successive window
        if random.random() >= successive_window_chance**(window_idx):
            window.location = (0, 0, 1000)
        else:
            depth = random.uniform(-z_bound, 4) # fix for the perspective
            window.location = (random.uniform(-x_bound, x_bound) * depth/10,
                               random.uniform(-y_bound, y_bound) * depth/10,
                               depth)

            window.rotation_euler = (random.uniform(-np.pi * 0.28, np.pi * 0.28),
                                     random.uniform(-np.pi * 0.28,
                                                    np.pi * 0.28),
                                     random.uniform(-np.pi * 1, np.pi * 1))
        if window:
            # window.select_set(True)
            objects.append(window_names[window_idx])

    # randomize the occlusion objects
    for obj_idx in range(len(object_names)):
        obj = bpy.data.objects.get(object_names[obj_idx])

        # move the unused objects out of the camera view
        if random.random() >= chance_of_occlusion:
            obj.location = (0, 0, 1000)
        else:
            depth = random.uniform(-z_bound, 2) # fix for the perspective
            obj.location = (random.uniform(-x_bound, x_bound) * depth/10,
                            random.uniform(-y_bound, y_bound) * depth/10,
                            depth)

            obj.rotation_euler = (random.uniform(-np.pi * 1, np.pi * 1),
                                  random.uniform(-np.pi * 1, np.pi * 1),
                                  random.uniform(-np.pi * 1, np.pi * 1))
        if obj:
            # obj.select_set(True)
            objects.append(object_names[obj_idx])
    
    # # make if keyframes
    # for obj in objects:
    #     bpy.data.objects[obj].keyframe_insert(data_path="location", index=img_idx)
    #     bpy.data.objects[obj].keyframe_insert(data_path="rotation_euler", index=img_idx)

    # randomize the lights
    light = bpy.data.objects.get("Light")
    light_data = light.data
    light.location = (random.uniform(-10, 10),
                      random.uniform(-10, 10),
                      random.uniform(5, 15))
    light_data.use_shadow = True
    light_data.energy = random.uniform(1000, 10000)

    # Set the output file names
    TrainingFile.file_slots[0].path = f"{img_idx:05}"
    SegmentedFile.file_slots[0].path = f"{img_idx:05}"
    # https://www.youtube.com/watch?v=xeprI8hJAH8
    # Render the scene and save the images
    bpy.ops.render.render(write_still=False)

print("Rendering done")

#----------------------------------------------------------
# go to each image and remove the trailing zeroes "*0000.png" -> ".png"
import os
for file in os.listdir(images_folder):
    if file.endswith(".png") and len(file) > 9:
        os.rename(images_folder+f"{file}", images_folder+f"{file[:-8]}.png")
for file in os.listdir(segmented_folder):
    if file.endswith(".png") and len(file) > 9:
        os.rename(segmented_folder+f"{file}", segmented_folder+f"{file[:-8]}.png")

print("Removing trailing zeroes done")