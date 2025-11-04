import os
import shutil
import numpy as np
from PIL import Image
import re

def dp(*args, **kwargs):
    kwargs['flush'] = True
    print(*args, **kwargs)
            
def CombineImages(pred, label, rgb):
    pred = pred.detach().cpu().numpy().squeeze()
    label = label.detach().cpu().numpy().squeeze()
    rgb = rgb.detach().cpu().numpy()
    
    pred_3ch = np.stack([pred, pred, pred], axis=0)
    label_3ch = np.stack([label, label, label], axis=0)

    # Concatenate images horizontally
    combined_image_np = np.concatenate((pred_3ch, label_3ch, rgb), axis=1)
    combined_image_np = (np.clip(combined_image_np, 0, 1)*255).astype(np.uint8)
    combined_image_np = combined_image_np.transpose(1, 2, 0)

    return combined_image_np


# def rename_files_in_folder(folder_path):
#     """
#     Rename files by removing the prefix and keeping only the number and extension.
#     Example: rgb001.png -> 001.png, mask001.png -> 001.png
#     """
#     if not os.path.exists(folder_path):
#         print(f"Folder does not exist: {folder_path}")
#         return
    
#     for filename in os.listdir(folder_path):
#         # Extract the numeric part from the filename
#         match = re.search(r'\d+', filename)
#         if match:
#             number = match.group()
#             # Get the file extension
#             _, ext = os.path.splitext(filename)
#             # Create new filename with just the number
#             new_filename = f"{number}{ext}"
            
#             old_path = os.path.join(folder_path, filename)
#             new_path = os.path.join(folder_path, new_filename)
            
#             # Rename the file
#             os.rename(old_path, new_path)
#             print(f"Renamed: {filename} -> {new_filename}")






# if __name__ == "__main__":
#     # # Update these paths to your actual directories
    # base_path = "/home/hkortus/scratch/window_images/"
    
    # images_path = base_path + "images/"
    # segmented_path = base_path + "segmented/"
    
    # print("Renaming files in images folder...")
    # rename_files_in_folder(images_path)
    
    # print("\nRenaming files in segmented folder...")
    # rename_files_in_folder(segmented_path)
    
    # print("\nDone!")