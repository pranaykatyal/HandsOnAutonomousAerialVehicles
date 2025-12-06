from splat_render import SplatRenderer
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pyquaternion import Quaternion
from enum import Enum, auto

from control import QuadrotorController
from quad_dynamics import model_derivative
import tello
from navigation import goToWaypoint, clear_old_imgs, timecounter
from flow import RaftFlow
from video_gen import create_overlay_animation_video

def save_comparison_with_iou(predicted_mask, gt_mask,img, iou, output_path, folder_name):
    """
    Save a side-by-side comparison of predicted mask and ground truth mask with IoU overlay.
    
    Parameters:
    - predicted_mask: Predicted binary mask (grayscale)
    - gt_mask: Ground truth binary mask (grayscale)
    - iou: IoU score to display
    - output_path: Path to save the comparison image
    - folder_name: Name of the sequence folder for title
    """
    
    # Ensure masks are binary for visualization
    # pred_viz = ((predicted_mask > 0) * 255).astype(np.uint8)
    # gt_viz = ((gt_mask > 0) * 255).astype(np.uint8)
    
    # Convert to BGR for color text
    pred_bgr = cv2.cvtColor(predicted_mask, cv2.COLOR_GRAY2BGR)
    gt_bgr = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)
    img = img.astype(np.uint8)

    # Add text labels to each mask
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    color = (0, 255, 0)  # Green text
    
    # Add "Predicted" label
    cv2.putText(pred_bgr, 'Predicted', (10, 30), font, font_scale, color, thickness)
    
    # Add "Ground Truth" label
    cv2.putText(gt_bgr, 'Ground Truth', (10, 30), font, font_scale, color, thickness)
    
    # Stack horizontally
    comparison = np.hstack([img,gt_bgr,pred_bgr ])
    
    # Add IoU text at the bottom center
    h, w = comparison.shape[:2]
    iou_text = f'IoU: {iou:.4f}'
    text_size = cv2.getTextSize(iou_text, font, 1.2, 3)[0]
    text_x = (w - text_size[0]) // 2
    text_y = h - 20
    
    # Add black background rectangle for text
    rect_margin = 10
    cv2.rectangle(comparison, 
                  (text_x - rect_margin, text_y - text_size[1] - rect_margin),
                  (text_x + text_size[0] + rect_margin, text_y + rect_margin),
                  (0, 0, 0), -1)
    
    # Add IoU text
    cv2.putText(comparison, iou_text, (text_x, text_y), font, 1.2, (0, 255, 255), 3)
    
    # Add title at the top
    title = f'Sequence: {folder_name}'
    title_size = cv2.getTextSize(title, font, 0.7, 2)[0]
    title_x = (w - title_size[0]) // 2
    
    # Add black background for title
    cv2.rectangle(comparison,
                  (title_x - rect_margin, 5),
                  (title_x + title_size[0] + rect_margin, 5 + title_size[1] + rect_margin),
                  (0, 0, 0), -1)
    
    cv2.putText(comparison, title, (title_x, 5 + title_size[1]), font, 0.7, (255, 255, 255), 2)
    
    # Save the comparison image
    cv2.imwrite(output_path, comparison)
    
    return comparison

def calculate_iou(mask1, mask2):
    """Calculate Intersection over Union between two binary masks"""
    # Ensure masks are binary
    mask1_binary = (mask1 > 0).astype(np.uint8)
    mask2_binary = (mask2 > 0).astype(np.uint8)
    
    # Calculate intersection and union
    intersection = np.logical_and(mask1_binary, mask2_binary).sum()
    union = np.logical_or(mask1_binary, mask2_binary).sum()
    
    # Avoid division by zero
    if union == 0:
        return 0.0
    
    iou = intersection / union
    return iou
    

def analyze_sequence_folders(base_path='/home/hkortus/scratch/Outputs/Sequences/', flow_model=None):
    """
    Iterate through all folders in the base path, select frame_00.png and frame_02.png,
    and calculate displacement using optical flow.
    
    Parameters:
    - base_path: Path to the directory containing sequence folders
    - flow_model: RaftFlow instance for flow calculation
    
    Returns:
    - results: Dictionary with folder names as keys and (ey, ez) tuples as values
    """
    import os
    import cv2
    
    if flow_model is None:
        flow_model = RaftFlow(
            model_pth='/home/hkortus/RBE595/HandsOnAutonomousAerialVehicles/YourDirectoryID_p4/RAFT/models/raft-things.pth',
            alternative_corr=True,
            mask_threshold=.35
        )
    
    results = {}
    
    # Check if base path exists
    if not os.path.exists(base_path):
        print(f"Error: Base path does not exist: {base_path}")
        return results
    
    # Iterate through all subdirectories
    for folder_name in sorted(os.listdir(base_path)):
        print("evaluateing folder")
        folder_path = os.path.join(base_path, folder_name)
        
        # Skip if not a directory
        if not os.path.isdir(folder_path):
            continue
        
        # Construct paths to frame_00.png and frame_02.png
        frame_00_path = os.path.join(folder_path, 'frame_00.png')
        frame_02_path = os.path.join(folder_path, 'frame_03.png')
        
        # Check if both frames exist
        if not os.path.exists(frame_00_path):
            print(f"Warning: {folder_name} - frame_00.png not found")
            continue
        
        if not os.path.exists(frame_02_path):
            print(f"Warning: {folder_name} - frame_03.png not found")
            continue
        
        # Load images
        img1 = cv2.imread(frame_00_path)
        img2 = cv2.imread(frame_02_path)
        
        if img1 is None or img2 is None:
            print(f"Warning: {folder_name} - Failed to load images")
            continue
        
        # Calculate displacement
        try:
            ey, ez, predicted_mask = flow_model.get_displacement_from_img_pair(img1, img2)

            if ey is None or ez == 0 or ez is None or ey == 0:
                print("skipping")
                continue
            
            # Load ground truth mask - mask_03.png since we used frame_03
            gt_mask_path = os.path.join(folder_path, 'mask_03.png')
            
            if not os.path.exists(gt_mask_path):
                print(f"Warning: {folder_name} - Ground truth mask not found at {gt_mask_path}")
                results[folder_name] = (ey, ez, None)
                continue
            
            gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
            print(f'compareing gt_mask og shape {gt_mask.shape}, min {gt_mask.min()} max {gt_mask.max()}')

            if gt_mask is None:
                print(f"Warning: {folder_name} - Failed to load ground truth mask")
                results[folder_name] = (ey, ez, None)
                continue
            
            # Invert ground truth mask (assuming black is hole, white is background)
            gt_mask = cv2.bitwise_not(gt_mask)
            gt_mask = cv2.normalize(gt_mask.astype(np.uint8), None, 0, 1, cv2.NORM_MINMAX)
            # Ensure predicted mask is grayscale if it's not already
            if len(predicted_mask.shape) == 3:
                predicted_mask = cv2.cvtColor(predicted_mask, cv2.COLOR_BGR2GRAY)
            
            # Get original image dimensions to use as reference
            target_height, target_width = img1.shape[:2]
            
            # Resize both masks to match the original image dimensions
            if predicted_mask.shape != (target_height, target_width):
                predicted_mask = cv2.resize(predicted_mask, (target_width, target_height), 
                                           interpolation=cv2.INTER_NEAREST)
            
            if gt_mask.shape != (target_height, target_width):
                gt_mask = cv2.resize(gt_mask, (target_width, target_height), 
                                    interpolation=cv2.INTER_NEAREST)
            
            if img2.shape != (target_height, target_width):
                img2 = cv2.resize(img2, (target_width, target_height), 
                    interpolation=cv2.INTER_NEAREST)
            # Verify dimensions match
            assert predicted_mask.shape == gt_mask.shape, \
                f"Mask dimensions don't match: predicted {predicted_mask.shape} vs gt {gt_mask.shape}"
            
            # Save comparison image for debugging
            comparison_dir = os.path.join(folder_path, 'comparison')
            os.makedirs(comparison_dir, exist_ok=True)
            predicted_mask_norm = cv2.normalize(predicted_mask.astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX)
            gt_mask_norm = cv2.normalize(gt_mask.astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX)
            # Create side-by-side comparison
            comparison = np.hstack([
                cv2.cvtColor(predicted_mask_norm, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(gt_mask_norm, cv2.COLOR_GRAY2BGR)
            ])
            cv2.imwrite(os.path.join(comparison_dir, 'mask_comparison.png'), comparison)

            print(f'compareing predicted_mask snape {predicted_mask.shape}, min {predicted_mask.min()} max {predicted_mask.max()}')
            print(f'compareing gt_mask snape {gt_mask.shape}, min {gt_mask.min()} max {gt_mask.max()}')

            # Calculate IoU
            iou = calculate_iou(predicted_mask, gt_mask)
            
            # Save comparison image for debugging
            comparison_dir = os.path.join(folder_path, 'comparison')
            os.makedirs(comparison_dir, exist_ok=True)
            
            # Create and save enhanced comparison with IoU
            comparison_path = os.path.join(comparison_dir, 'mask_comparison_with_iou.png')
            save_comparison_with_iou(predicted_mask_norm, gt_mask_norm,img2, iou, comparison_path, folder_name)

            results[folder_name] = (ey, ez, iou)
            print(f"{folder_name}: ey={ey:.2f}px, ez={ez:.2f}px, IoU={iou:.4f}")
            
        except Exception as e:
            print(f"Error processing {folder_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nProcessed {len(results)} folders successfully")
    
    # Print summary statistics
    ious = [iou for _, _, iou in results.values() if iou is not None]
    if ious:
        print(f"\nIoU Statistics:")
        print(f"  Mean IoU: {np.mean(ious):.4f}")
        print(f"  Median IoU: {np.median(ious):.4f}")
        print(f"  Min IoU: {np.min(ious):.4f}")
        print(f"  Max IoU: {np.max(ious):.4f}")
    
    return results


if __name__ == "__main__":
    config_path = "../data/p4_colmap_nov6_1000_splat/p4_colmap_nov6_1000/splatfacto/2025-11-06_161816/config.yml"
    json_path = "../render_settings/render_settings.json"

    # renderer = SplatRenderer(config_path, json_path)
    analyze_sequence_folders()
    # main(renderer)
    print('creating videos')
    succ = create_overlay_animation_video(overlay_dir='/home/hkortus/RBE595/HandsOnAutonomousAerialVehicles/YourDirectoryID_p4/Code/run',
                                   frames_dir='/home/hkortus/RBE595/HandsOnAutonomousAerialVehicles/YourDirectoryID_p4/Code/imgs', 
                                   output_dir='/home/hkortus/RBE595/HandsOnAutonomousAerialVehicles/YourDirectoryID_p4/Code',
                                   fps=35)
    print(f'succses = {succ}')
# /data/p4_colmap_nov6_1000_splat/p4_colmap_nov6_1000/splatfacto/2025-11-06_161816/config.yml