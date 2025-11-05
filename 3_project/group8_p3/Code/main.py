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

################################################
#### Navigation Function ########################
################################################
def goToWaypoint(currentPose, targetPose, velocity=0.1):
    """
    Navigate quadrotor to a target waypoint
    
    Parameters:
    - currentPose: Dictionary with keys:
        'position': [x, y, z] in NED frame (meters)
        'rpy': [roll, pitch, yaw] in radians
    - targetPose: [x, y, z] target position in NED frame (meters)
    - velocity: cruise velocity (m/s), default 1.0
    
    Returns:
    - newPose: Dictionary with updated 'position' and 'rpy'
    """
    
    dt = 0.01  # 10ms timestep
    tolerance = 0.1  # 10cm tolerance
    max_time = 30.0  # Maximum 30 seconds
    
    # Initialize controller
    controller = QuadrotorController(tello)
    param = tello
    
    # Extract current state
    pos = np.array(currentPose['position'])
    rpy = np.array(currentPose['rpy'])  # roll, pitch, yaw in radians
    
    # Initialize velocities to zero (starting from hover)
    vel = np.zeros(3)
    pqr = np.zeros(3)
    
    # Convert roll, pitch, yaw to quaternion
    roll, pitch, yaw = rpy
    quat = Quaternion(axis=[0, 0, 1], radians=yaw) * \
           Quaternion(axis=[0, 1, 0], radians=pitch) * \
           Quaternion(axis=[1, 0, 0], radians=roll)
    
    # Build state vector [x, y, z, vx, vy, vz, qx, qy, qz, qw, p, q, r]
    current_state = np.concatenate([pos, vel, [quat.x, quat.y, quat.z, quat.w], pqr])
    
    target_position = np.array(targetPose)
    
    # Calculate distance and estimated time
    distance = np.linalg.norm(target_position - pos)
    estimated_time = min(distance / velocity * 2.0, max_time)
    
    print(f"  Navigating: {pos} → {target_position}")
    print(f"  Distance: {distance:.2f}m, Est. time: {estimated_time:.1f}s")
    
    # Check if already at target
    if distance < tolerance:
        print("  Already at target!")
        return {'position': pos, 'rpy': rpy}
    
    # Generate trajectory
    num_points = int(estimated_time / dt)
    time_points = np.linspace(0, estimated_time, num_points)
    
    # Create trajectory with trapezoidal velocity profile
    direction = target_position - pos
    unit_direction = direction / distance
    
    trajectory_points = []
    velocities = []
    accelerations = []
    
    accel_time = min(1.0, estimated_time * 0.25)
    decel_time = accel_time
    cruise_time = estimated_time - accel_time - decel_time
    
    cruise_vel = min(velocity, distance / (0.5 * accel_time + cruise_time + 0.5 * decel_time))
    
    for t in time_points:
        if t <= accel_time:
            # Acceleration phase
            vel_mag = (cruise_vel / accel_time) * t
            acc_mag = cruise_vel / accel_time
            progress = 0.5 * (cruise_vel / accel_time) * t * t / distance
        elif t <= accel_time + cruise_time:
            # Cruise phase
            vel_mag = cruise_vel
            acc_mag = 0.0
            progress = (0.5 * cruise_vel * accel_time + cruise_vel * (t - accel_time)) / distance
        else:
            # Deceleration phase
            t_decel = t - accel_time - cruise_time
            vel_mag = cruise_vel - (cruise_vel / decel_time) * t_decel
            acc_mag = -cruise_vel / decel_time
            progress = (0.5 * cruise_vel * accel_time + cruise_vel * cruise_time + 
                      cruise_vel * t_decel - 0.5 * (cruise_vel / decel_time) * t_decel * t_decel) / distance
        
        progress = np.clip(progress, 0.0, 1.0)
        position = pos + progress * direction
        vel_vec = vel_mag * unit_direction
        acc_vec = acc_mag * unit_direction
        
        trajectory_points.append(position)
        velocities.append(vel_vec)
        accelerations.append(acc_vec)
    
    trajectory_points = np.array(trajectory_points)
    velocities = np.array(velocities)
    accelerations = np.array(accelerations)
    
    # Set trajectory in controller
    controller.set_trajectory(trajectory_points, time_points, velocities, accelerations)
    
    # Simulation loop
    state = current_state.copy()
    
    for i, t in enumerate(time_points):
        # Compute control input
        control_input = controller.compute_control(state, t)
        
        # Check if reached
        current_pos = state[0:3]
        error = np.linalg.norm(current_pos - target_position)
        if error < tolerance and t > 1.0:
            print(f"  ✓ Reached at t={t:.2f}s, error={error:.3f}m")
            state_final = state
            break
        
        # Integrate dynamics
        if i < len(time_points) - 1:
            sol = solve_ivp(
                lambda t, X: model_derivative(t, X, control_input, param),
                [t, t + dt],
                state,
                method='RK45',
                max_step=dt
            )
            state = sol.y[:, -1]
            state_final = state
    else:
        # Loop completed without break
        state_final = state
        error = np.linalg.norm(state_final[0:3] - target_position)
        print(f"  Final error: {error:.3f}m")
    
    # Extract final pose
    final_pos = state_final[0:3]
    final_quat = Quaternion(state_final[9], state_final[6], state_final[7], state_final[8])  # w, x, y, z
    final_ypr = final_quat.yaw_pitch_roll  # Returns [yaw, pitch, roll]
    final_rpy = np.array([final_ypr[2], final_ypr[1], final_ypr[0]])  # [roll, pitch, yaw]
    
    newPose = {
        'position': final_pos,
        'rpy': final_rpy
    }
    
    return newPose


################################################
#### Main Function ##############################
################################################
def main(renderer):
    #set up model 
    segmentor = Window_Segmentaion(torch_network=Network,
                                   model_path=TRAINED_MODEL_PATH,
                                   model_thresh=.98,
                                   in_ch=3,out_ch=1,img_h=256,img_w=256)



    # Create log directory if it doesn't exist
    import os
    os.makedirs('./log', exist_ok=True)
    
    # Initialize pose - Position: x, y, z in meters | Orientation: roll, pitch, yaw in radians
    currentPose = {
        'position': np.array([0.0, 0.0, -0.2]),  # NED origin
        'rpy': np.radians([0.0, 0.0, 0.0])      # Orientation origin
    }
    
    numWindows = 3

    # iterate through windows one by one
    for windowCount in range(numWindows):
        print(f"\n=== Window {windowCount + 1}/{numWindows} ===")
        
        # Render the frame at the current pose
        color_image, depth_image, metric_depth = renderer.render(
            currentPose['position'], 
            currentPose['rpy']
        )

        # brightness_increase = 55
        # boosted_color_image = np.clip((color_image.astype(int)*1.85) + brightness_increase, 0, 255).astype(np.uint8)
        segmented = segmentor.get_pred(color_image)
        print(f'segmentd stats {segmented.max()}, main:{segmented.min()}')
        segmented = cv2.normalize(segmented, None, 0, 255, cv2.NORM_MINMAX)
        segmented = segmented.astype(np.uint8)
        cv2.imwrite(f'segmentaion{windowCount}.png', segmented)
        #####################################################
        ### DETECT THE WINDOW AND NAVIGATION THROUGH IT #####
        #####################################################
        
        ## Segment the window from color image
        ## Feel free to use metric_depth or depth_image as well  
        ## Write your own logic to go through the window
        ## Example: Use windowMask to extract window region,
        ## compute centroid, use depth to estimate distance,
        ## then navigate through it

        ## you are free to use any method you like to navigate through the window
        ## you can tweak the trajectory planner as well if you want
        
        # Example target (you should compute this from windowMask and depth)
        # targetPose = np.array([1.0 * (windowCount + 1), 0.0, 0.0])
        
        # Navigate to the target waypoint
        # currentPose = goToWaypoint(currentPose, targetPose, velocity=1.0)
        
        # Save the color image
        color_image_bgr = cv2.cvtColor(color_image,cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'rendered_frame_window_{windowCount}.png', color_image_bgr)

        # Save the depth image (normalized for visualization)
        depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = depth_normalized.astype(np.uint8)
        cv2.imwrite(f'depth_frame_window_{windowCount}.png', depth_normalized)

        print(f'Saved frame for window {windowCount} at position {currentPose["position"]}')
       
        return False

if __name__ == "__main__":
    config_path = "../data/washburn-env6-itr0-1fps/washburn-env6-itr0-1fps_nf_format/splatfacto/2025-03-06_201843/config.yml"
    json_path = "../render_settings/render_settings_2.json"

    renderer = SplatRenderer(config_path, json_path)
    main(renderer)
