import numpy as np
from scipy.integrate import solve_ivp
from pyquaternion import Quaternion
from control import QuadrotorController
from quad_dynamics import model_derivative
import tello
import cv2

def goToWaypoint(currentPose, targetPose, targetOrientation=None, velocity=0.1):
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
    """
    
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
    tolerance = 0.01  # 10cm tolerance
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
    
    # Calculate distance and estimated time
    distance = np.linalg.norm(target_position - pos)
    estimated_time = min(distance / velocity * 2.0, max_time)
    
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
    
    for i, t in enumerate(time_points):
        # Compute control input
        control_input = controller.compute_control(state, t)
        
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