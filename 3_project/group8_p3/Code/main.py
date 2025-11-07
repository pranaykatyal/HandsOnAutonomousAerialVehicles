from splat_render import SplatRenderer
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pyquaternion import Quaternion
from control import QuadrotorController
from quad_dynamics import model_derivative
import tello
from window_segmentation.window_segmentation import Window_Segmentaion
from window_segmentation.network import Network
from params import *
from window_detector import WindowDetector
from orientation_navigation import navigate_with_orientation_correction
from navigation import goToWaypoint, navigate_through_window
import os
import shutil
import glob

################################################
#### Main Function ##############################
################################################
def main(renderer):
    """Main racing loop - navigate through multiple windows"""
    
    # Set up segmentation model
    segmentor = Window_Segmentaion(
        torch_network=Network,
        model_path=TRAINED_MODEL_PATH,
        model_thresh=0.90,
        in_ch=3, 
        out_ch=1, 
        img_h=256, 
        img_w=256
    )
    
    # Set up window detector (adjust dimensions based on render_settings_2.json)
    detector = WindowDetector(image_width=256, image_height=256)

    # Create log directory and clean old image files

    # Create log directory if it doesn't exist
    os.makedirs('./log', exist_ok=True)

    # Clear only image files from previous runs
    old_images = glob.glob('./log/*.png')
    if old_images:
        print(f"Cleaning {len(old_images)} old image files...")
        for img_file in old_images:
            try:
                os.remove(img_file)
            except Exception as e:
                print(f"Warning: Could not remove {img_file}: {e}")
    else:
        print("No old image files to clean")
    
    # Initialize pose - NED frame
    currentPose = {
        'position': np.array([0.0, 0.0, -0.2]),  # Start slightly elevated
        'rpy': np.radians([0, 0.0, 0.0])       # Level orientation
    }
    # x = renderer.render(currentPose['position'], currentPose['rpy'])
    # cv2.imwrite('bla.png', x)
    # exit()
    
    numWindows = 3
    successful_windows = 0
    
    # Main racing loop
    for windowCount in range(numWindows):
        success, currentPose = navigate_with_orientation_correction(
            renderer=renderer,
            currentPose=currentPose,
            segmentor=segmentor,
            detector=detector,
            windowCount=windowCount,
            max_iterations=60
        )
        
        if success:
            successful_windows += 1
            print(f"\n Window {windowCount + 1} PASSED")
        else:
            print(f"\n Window {windowCount + 1} FAILED")
            # Optional: decide whether to continue or abort
            # break  # Uncomment to stop on first failure
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"RACE COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully passed: {successful_windows}/{numWindows} windows")
    print(f"Final position: {currentPose['position']}")
    
    return successful_windows == numWindows


if __name__ == "__main__":
    # Update these paths to match your setup
    config_path = "../data/washburn-env6-itr0-1fps/washburn-env6-itr0-1fps_nf_format/splatfacto/2025-03-06_201843/config.yml"
    json_path = "../render_settings/render_settings_2.json"
    
    renderer = SplatRenderer(config_path, json_path)
    success = main(renderer)
    
    if success:
        print("\n All windows cleared successfully!")
    else:
        print("\n  Some windows were not cleared")