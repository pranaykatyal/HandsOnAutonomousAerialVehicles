import numpy as np
from scipy.integrate import solve_ivp
from pyquaternion import Quaternion
from control import QuadrotorController
from quad_dynamics import model_derivative
from dataclasses import dataclass

import tello
import cv2
import os

# Global frame counter for unique naming
_frame_counter = 0


@dataclass
class timecounter:
    time = 0

    def increment_time(self, increment):
        self.time += increment

    def reset_time(self):
        self.time = 0

    def get_time(self):
        return self.time


# Time = timecounter()


def goToWaypoint(currentPose, targetPose, targetOrientation=None, velocity=0.1, 
                 renderer=None, segmentor=None,  window_id=0, iteration_id=0, save_every=5, Time=None):
    """
    Navigate quadrotor to a target waypoint
    
    Parameters:
    - currentPose: Dictionary with keys:
        'position': [x, y, z] in NED frame (meters)
        'rpy': [roll, pitch, yaw] in radians
    - targetPose: Dictionary OR array
        If dict: {'position': [x,y,z], 'rpy': [r,p,y]} - control both
        If array: [x, y, z]
    - targetOrientation: Optional target orientation [roll, pitch, yaw] in radians
    - velocity: cruise velocity (m/s), default 0.1
    - renderer: SplatRenderer instance for capturing frames (optional)
    - segmentor: Window_Segmentaion instance for segmentation (optional)
    - window_id: Current window number for frame naming
    - iteration_id: Current iteration number for frame naming
    - save_every: Save every Nth frame (default: 5)
    """
    
    global _frame_counter
    
    # Handle both formats for backward compatibility
    if isinstance(targetPose, dict):
        target_position = np.array(targetPose['position'])
        target_orientation = np.array(targetPose['rpy'])
        control_orientation = True
    else:
        # Old format: just position
        target_position = np.array(targetPose)
        # Use targetOrientation parameter if provided, otherwise use zeros
        if targetOrientation is not None:
            target_orientation = np.array(targetOrientation)
        else:
            target_orientation = np.zeros(3)
        control_orientation = False
    
    dt = 0.01  # 10ms timestep
    tolerance = 0.001  # 10cm tolerance
    max_time = 10.0  # Maximum 30 seconds
    
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
    
    # Calculate distance and estimated time
    distance = np.linalg.norm(target_position - pos)
    estimated_time = min(distance / velocity * 2.0, max_time)
    if Time:
        Time.increment_time(estimated_time)

    print(f"  Navigating: {pos} to {target_position}")
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
    controller.set_trajectory(trajectory_points, time_points, velocities, accelerations, target_orientation)
    
    # Simulation loop
    state = current_state.copy()
    local_frame_count = 0
    
    for i, t in enumerate(time_points):
        # Compute control input
        control_input = controller.compute_control(state, t)
        
        # Save intermediate frames if renderer is provided
        if renderer is not None and segmentor is not None and (local_frame_count % save_every == 0):
            current_pos = state[0:3]
            current_quat = Quaternion(state[9], state[6], state[7], state[8])  # w, x, y, z
            current_ypr = current_quat.yaw_pitch_roll
            current_rpy = np.array([current_ypr[2], current_ypr[1], current_ypr[0]])
            
            # Render frame
            color_image, depth_image, metric_depth = renderer.render(current_pos, current_rpy)
            segmented = segmentor.get_pred(color_image)
            segmented = cv2.normalize(segmented, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Save with unique frame counter
            frame_prefix = f'./log/window_{window_id}_iter_{iteration_id:02d}_frame_{_frame_counter:04d}'
            cv2.imwrite(f'{frame_prefix}_rgb.png', cv2.flip(cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR), 0))
            cv2.imwrite(f'{frame_prefix}_segmentation.png', cv2.flip(segmented, 0))
            _frame_counter += 1
        
        local_frame_count += 1
        
        # Check if reached
        current_pos = state[0:3]
        error = np.linalg.norm(current_pos - target_position)
        if error < tolerance and t > 1.0:
            print(f" Reached at t={t:.2f}s, error={error:.3f}m")
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
        print(f" WARNING: path did not fully complete! Final error: {error:.3f}m")
    
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




def reset_frame_counter():
    """Reset the global frame counter (call at start of simulation)"""
    global _frame_counter
    _frame_counter = 0