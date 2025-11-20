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
from navigation import goToWaypoint, clear_old_imgs
from flow import RaftFlow

class State(Enum):
    START = auto()
    SENSE = auto()
    SERVO = auto()
    FLY_THROUGH = auto()
    FINISHED = auto()




def main(renderer):
    # Create log directory if it doesn't exist
    import os
    os.makedirs('./log', exist_ok=True)
    clear_old_imgs()
    # Initialize pose - Position: x, y, z in meters | Orientation: roll, pitch, yaw in radians
    currentPose = {
        'position': np.array([-0.10, -0.20, 0.10]),  # NED origin
        #'position': np.array([0.20, -0.20, 0.0]),  # NED origin
        'rpy': np.radians([0, 0.0, 0.0])      # Orientation origin
    }
    

    flow = RaftFlow(model_pth='/home/hkortus/RBE595/HandsOnAutonomousAerialVehicles/YourDirectoryID_p4/RAFT/models/raft-things.pth',
                    alternative_corr=True)
    
    state = State.START
    # renderer
    color_image, depth_image, metric_depth = renderer.render(
        currentPose['position'], 
        currentPose['rpy'])
    img1= None
    img2 = None
    CENTER_THRESHOLD_PX = 20
    frame_cnt = 0

    for n in range(200):
        targetPose = None
        match state:
            case State.START:
                #move a little bit to generate some flow
                targetPose = np.array([0.1, -0.2, 0.00014])    
                state = State.SENSE

            case State.SENSE:
                if img1 is None or img2 is None:
                    print("No images available for flow calculation")
                    break

                ey, ez = flow.get_displacement_from_img_pair(img1, img2)

                if abs(ey) <= CENTER_THRESHOLD_PX and abs(ez) <= CENTER_THRESHOLD_PX:
                    state = State.FLY_THROUGH
                else:
                    state = State.SERVO
            case State.SERVO:
                # Reposition with intermediate frame capture
                ky = .0006 
                kz = .0006
                ctrl_y = ey * ky
                ctrl_z = ez * kz
                print(f'repositioning - ey: {ey}, ez: {ez}, ctrl_y {ctrl_y}, ctrl_z{ctrl_z}')

                targetPose = currentPose['position'].copy()
                targetPose[1] += ctrl_y
                targetPose[2] += ctrl_z
                state = State.SENSE

            case State.FLY_THROUGH:
                print("!!!!!!!!!FLYTING THOUGH GAP!!!!!!")
                targetPose = currentPose['position'].copy()
                targetPose[0] += 0.5
                state = State.FINISHED

            case State.FINISHED:
                print("done with run!!!!")
                break

# Only call goToWaypoint if targetPose was set
        if targetPose is not None:
            currentPose, img1, img2 = goToWaypoint(currentPose=currentPose, 
                                                   targetPose=targetPose,
                                                   flow_image_distance=0.01, 
                                                   velocity=0.02, 
                                                   renderer=renderer,
                                                   save_every=10, 
                                                   iteration_id=frame_cnt)
            
            frame_cnt = frame_cnt + 1
            print(f"frame count: {frame_cnt}")
            #if our path is so small that we cant resolve the flow, dont move and fly streight
            if img1 is None or img2 is None:
                print("this is a shitty patch and will bite me in the ass later")
                state = State.FLY_THROUGH


if __name__ == "__main__":
    config_path = "../data/p4_colmap_nov6_1000_splat/p4_colmap_nov6_1000/splatfacto/2025-11-06_161816/config.yml"
    json_path = "../render_settings/render_settings.json"

    renderer = SplatRenderer(config_path, json_path)
    main(renderer)
# /data/p4_colmap_nov6_1000_splat/p4_colmap_nov6_1000/splatfacto/2025-11-06_161816/config.yml