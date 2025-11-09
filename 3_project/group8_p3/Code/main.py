from splat_render import SplatRenderer
import numpy as np
import cv2
# import matplotlib.pyplot as plt
# from scipy.integrate import solve_ivp
# from pyquaternion import Quaternion
# from control import QuadrotorController
# from quad_dynamics import model_derivative
# import tello
from window_segmentation.window_segmentation import Window_Segmentaion
from window_segmentation.network import Network
from params import *
from window_detector import WindowDetector
# from orientation_navigation import navigate_with_orientation_correction
from navigation import goToWaypoint, reset_frame_counter
import os
import shutil
import glob



def init_log_dir():
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
    


################################################
#### Main Function ##############################
################################################
def main(renderer):
    init_log_dir()
    reset_frame_counter()  # Reset frame counter at start

    # Set up segmentation model
    segmentor = Window_Segmentaion(
        torch_network=Network,
        model_path=TRAINED_MODEL_PATH,
        model_thresh=0.70,
        in_ch=3, 
        out_ch=1, 
        img_h=256, 
        img_w=256
    )
    
    # Initialize pose - NED frame
    currentPose = {
        'position': np.array([0.0, 0.0, 0.0]),  # Start slightly elevated
        'rpy': np.radians([np.pi, 0.0, 0.0])       # Level orientation
    }
    
    # Initial movement with frame capture
    target_pos = currentPose['position'].copy()
    target_pos[0] += 0.1
    target_rpy = np.zeros_like(currentPose['rpy'])  # Maintain current orientation
    currentPose = goToWaypoint(currentPose, target_pos, target_rpy, velocity=0.02,
                               renderer=renderer, segmentor=segmentor, 
                               window_id=-1, iteration_id=0, save_every=50)

    color_image, depth_image, metric_depth = renderer.render(
                currentPose['position'], 
                currentPose['rpy']
                )
    numWindows = 3
    successful_windows = 0
    
    # Main racing loop
    for windowCount in range(numWindows):

        for n in range(ALIGNMENT_ATTEMPTS):
            # Render current view
            color_image, depth_image, metric_depth = renderer.render(
                currentPose['position'], 
                currentPose['rpy']
                )
            
            # Segment for viewing purposes only
            segmented = segmentor.get_pred(color_image)
            segmented = cv2.normalize(segmented, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)           
            
            # Save alignment iteration images
            iter_prefix = f'./log/window_{windowCount}_iter_{n:02d}_align'
            cv2.imwrite(f'{iter_prefix}_rgb.png', cv2.flip(cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR), 0))
            cv2.imwrite(f'{iter_prefix}_segmentation.png', cv2.flip(segmented, 0))
            
            # Get centroid of frame
            ex, ey, ez = segmentor.get_closest_frame(color_image, metric_depth)
            
            if np.abs(ey * ex) < WINDOW_THRESHOLD and np.abs(ez * ex) < WINDOW_THRESHOLD:
                print("!!! aligned, flyting though gate")
                if ex > APPROACH_DISTANCE:
                    print("\n Too far, going most of the way")
                    target_pos = currentPose['position'].copy()
                    target_pos[0] += (ex - APPROACH_DISTANCE)
                    target_rpy = np.zeros_like(currentPose['rpy'])
                    currentPose = goToWaypoint(currentPose, target_pos, target_rpy, velocity=0.1,
                                            renderer=renderer, segmentor=segmentor,
                                            window_id=windowCount, iteration_id=n, save_every=5)
                    
                else:
                    print('Close enough, fully flying though gate')
                    target_pos = currentPose['position'].copy()
                    target_pos[0] += ex
                    target_rpy = np.zeros_like(currentPose['rpy'])
                    currentPose = goToWaypoint(currentPose, target_pos, target_rpy, velocity=0.2,
                                            renderer=renderer, segmentor=segmentor,
                                            window_id=windowCount, iteration_id=n, save_every=5)
                    
                    print(f"Stage 1 complete. Position: {currentPose['position']}")
                    print("===== Window pass complete! =====")
                    print(f"Final position: {currentPose['position']}")
                    success = True
                    break
            else:

                # Reposition with intermediate frame capture
                print('repositioning')
                ky = .003 
                kz = .003
                ctrl_y = ey * ex * ky
                ctrl_z = ez * ex * kz
                
                target_pos = currentPose['position'].copy()
                target_pos[1] += -ctrl_y
                target_pos[2] += -ctrl_z
                target_rpy = np.zeros_like(currentPose['rpy'])
                currentPose = goToWaypoint(currentPose, target_pos, target_rpy, velocity=0.25,
                                          renderer=renderer, segmentor=segmentor,
                                          window_id=windowCount, iteration_id=n, save_every=10)
        else:
            print('ERROR!! FAILED TO ALIGN BODY TO FRAME')
            print('increase ALIGNMENT_ATTEMPTS, if this keeps happening, look into tuning')
            success = False

        if success:
            successful_windows += 1
            print(f"\n Window {windowCount + 1} PASSED")
        else:
            print(f"\n Window {windowCount + 1} FAILED")
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