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
from navigation import goToWaypoint
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
    


def fly_though_window(segmentor:Window_Segmentaion, windowCount:int=999) -> bool:
        for n in range(ALIGNMENT_ATTEMPTS):
            # rpy[0] = (rpy[0]+np.pi) % (np.pi * 2)
            color_image, depth_image, metric_depth = renderer.render(
                currentPose['position'], 
                currentPose['rpy']
                )
            # Segment for viewing porposes only, we never use segmented vareable
            segmented = segmentor.get_pred(color_image)
            segmented = cv2.normalize(segmented, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)           
            # Save images- 
            iter_prefix = f'./log/window_{windowCount}_iter_{n:02d}'
            cv2.imwrite(f'{iter_prefix}_rgb.png', cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f'{iter_prefix}_segmentation.png', segmented)
            #get centriod of frame
            ex, ey, ez = segmentor.get_closest_frame(color_image, metric_depth) #get the location error in px of the centroid of the nearest frame
            print(f'{windowCount}, iter{n} window ex {ex}, ey {ey}, ez {ez}')
            # kx = .001
            ky = .005 #TODO tune this
            kz = .005
            ctrl_y = ey * ex * ky #for y and z multiply the pixle error by how far away the object is with this scaleing term- we can mess with it, they should technically be the same
            ctrl_z = ez * ex * kz

            print(f'{windowCount}, iter{n} ctrl_y{ctrl_y}, ctrl_z{ctrl_z}')
            if np.abs(ey * ex) < WINDOW_THRESHOLD and np.abs(ez * ex) < WINDOW_THRESHOLD:
                #if we are within a threshold, fly through the frame, using the frames depth to figure out how far to fly
                print("!!!!! should be good to fly thogh frame now!!!!")
                print("flying though window")
                target_pos = currentPose['position'].copy()
                target_pos[0] += ex + .003 #add some tolerance
                target_rpy = np.zeros_like(currentPose['rpy']) # Maintain current orientation
                currentPose = goToWaypoint(currentPose, target_pos, target_rpy, velocity=0.03) #TODO make velocity value larget
                success = True
                break
            else:
                #if we still need to correct do that
                target_pos = currentPose['position'].copy()
                target_pos[1] += -ctrl_y #imo this is more inturtive than -=
                target_pos[2] += -ctrl_z
                target_rpy = np.zeros_like(currentPose['rpy']) # 
                currentPose = goToWaypoint(currentPose, target_pos, target_rpy, velocity=0.005) #TODO make velocity value larget
        else:
            print('ERROR!! FAILED TO ALIGN BODY TO FRAME')
            print('increse ALIGNMENT_ATTEMPTS, if this keeps happening, look into tuneing')
            return False

        print("done flying though frame:")
        print(currentPose['position'])
    


################################################
#### Main Function ##############################
################################################
def main(renderer):
    init_log_dir()

    # Set up segmentation model
    segmentor = Window_Segmentaion(
        torch_network=Network,
        model_path=TRAINED_MODEL_PATH,
        model_thresh=0.96,
        in_ch=3, 
        out_ch=1, 
        img_h=480, 
        img_w=640
    )
    # Initialize pose - NED frame
    currentPose = {
        'position': np.array([0.0, 0.0, 0.0]),  # Start slightly elevated
        'rpy': np.radians([np.pi, 0.0, 0.0])       # Level orientation
    }
    target_pos = currentPose['position'].copy()
    target_pos[0] += 0.1
    target_rpy = np.zeros_like(currentPose['rpy'])  # Maintain current orientation
    currentPose = goToWaypoint(currentPose, target_pos, target_rpy, velocity=0.02)

    color_image, depth_image, metric_depth = renderer.render(
                currentPose['position'], 
                currentPose['rpy']
                )
    numWindows = 3
    successful_windows = 0
    
    # Main racing loop
    for windowCount in range(numWindows):
        success = fly_though_window(segmentor=segmentor, windowCount=windowCount)
        if success:
            successful_windows += 1
            print(f"\n Window {windowCount + 1} PASSED")
        else:
            print(f"\n Window {windowCount + 1} FAILED")
            break  # Uncomment to stop on first failure
    
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