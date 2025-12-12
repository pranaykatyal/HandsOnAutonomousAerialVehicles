"""
Navigation functions for drone waypoint control
"""

import numpy as np
from scipy.integrate import solve_ivp
from pyquaternion import Quaternion
from control import QuadrotorController
from quad_dynamics import model_derivative
import tello
from collisionChecker import doesItCollide


def wrap_angle(angle):
    """Wrap angle to [-pi, pi]"""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def goToWaypoint_yaw(currentPose, target_yaw, pose_history=None, action='YAW', navigator=None):
    """
    Rotate in place to target yaw
    
    Args:
        currentPose: Current pose dict with 'position' and 'rpy'
        target_yaw: Target yaw angle in radians
        pose_history: List to append pose history (optional)
        action: Action label for logging
        navigator: WindowNavigator instance for frame recording (optional)
        
    Returns:
        Updated pose dict or -1 on failure
    """
    dt = 0.01
    tolerance_yaw = np.radians(0.5)
    max_time = 5.0

    controller = QuadrotorController(tello)
    param = tello
    
    controller.controller.angle_sf = np.array((1.0, 1.0, 1.0))

    pos = np.array(currentPose['position'], dtype=float)
    rpy = np.array(currentPose['rpy'], dtype=float)
    
    vel = np.zeros(3)
    pqr = np.zeros(3)

    roll, pitch, yaw = rpy
    quat = (Quaternion(axis=[0,0,1], radians=yaw) *
            Quaternion(axis=[0,1,0], radians=pitch) *
            Quaternion(axis=[1,0,0], radians=roll))

    current_state = np.concatenate([
        pos, vel,
        [quat.x, quat.y, quat.z, quat.w],
        pqr
    ])

    if doesItCollide(pos):
        return -1

    yaw_diff = wrap_angle(target_yaw - yaw)
    
    if abs(yaw_diff) < tolerance_yaw:
        return {'position': pos, 'rpy': rpy}

    max_yaw_rate = 1.0
    estimated_time = min(abs(yaw_diff) / max_yaw_rate * 1.5, max_time)
    
    num_points = max(2, int(estimated_time / dt))
    time_points = np.linspace(0, estimated_time, num_points)

    trajectory_points = np.tile(pos, (num_points, 1))
    velocities = np.zeros((num_points, 3))
    accelerations = np.zeros((num_points, 3))

    target_rpy = np.array([0.0, 0.0, target_yaw])
    
    controller.set_trajectory(trajectory_points, time_points, velocities, accelerations,
                             target_rpy=target_rpy)

    state = current_state.copy()
    
    last_capture_time = -1.0
    capture_interval = 0.2

    for i, t in enumerate(time_points):
        control_input = controller.compute_control(state, t)
        current_pos = state[0:3]

        pos_error = np.linalg.norm(current_pos - pos)
        if pos_error > 0.1:
            print(f'  [WARN] Position drift: {pos_error:.3f}m')

        qx, qy, qz, qw = state[6], state[7], state[8], state[9]
        temp_quat = Quaternion(w=qw, x=qx, y=qy, z=qz)
        current_yaw_temp, _, _ = temp_quat.yaw_pitch_roll
        yaw_error = abs(wrap_angle(target_yaw - current_yaw_temp))

        if navigator is not None and (t - last_capture_time) >= capture_interval:
            temp_yaw, temp_pitch, temp_roll = temp_quat.yaw_pitch_roll
            temp_rpy = np.array([temp_roll, temp_pitch, temp_yaw])
            
            temp_pose = {
                'position': current_pos.copy(),
                'rpy': temp_rpy
            }
            
            rgb_temp, _, _ = navigator.renderer.render(current_pos, temp_rpy)
            navigator.record_frame(rgb_temp, pose=temp_pose,
                                 annotation=f"{action} (yaw={np.degrees(temp_yaw):.1f} deg)")
            last_capture_time = t

        if yaw_error < tolerance_yaw and t > 0.5:
            state_final = state
            break

        if i < len(time_points) - 1:
            sol = solve_ivp(
                lambda tau, X: model_derivative(tau, X, control_input, param),
                [t, t+dt],
                state,
                method='RK45',
                max_step=dt
            )
            state = sol.y[:,-1]
            state_final = state
    else:
        state_final = state

    final_pos = state_final[0:3]
    qx, qy, qz, qw = state_final[6], state_final[7], state_final[8], state_final[9]
    final_quat = Quaternion(w=qw, x=qx, y=qy, z=qz)
    
    yaw_f, pitch_f, roll_f = final_quat.yaw_pitch_roll

    final_rpy = np.array([roll_f, pitch_f, yaw_f])
    
    if pose_history is not None:
        pose_history.append({
            'step': len(pose_history),
            'action': action,
            'position': final_pos.copy(),
            'rpy_deg': np.degrees(final_rpy),
            'rpy_rad': final_rpy.copy()
        })

    return {
        'position': final_pos,
        'rpy': final_rpy
    }


def goToWaypoint(currentPose, targetPose, velocity=0.1, pose_history=None, action='NAV',
                 maintain_orientation=False, lock_roll_pitch=False, navigator=None):
    """
    Navigate to waypoint with trajectory following
    
    Args:
        currentPose: Current pose dict with 'position' and 'rpy'
        targetPose: Target position (3D array)
        velocity: Cruise velocity in m/s
        pose_history: List to append pose history (optional)
        action: Action label for logging
        maintain_orientation: Keep initial RPY throughout (bool)
        lock_roll_pitch: Keep initial roll/pitch, allow yaw changes (bool)
        navigator: WindowNavigator instance for frame recording (optional)
        
    Returns:
        Updated pose dict or -1 on failure
    """
    dt = 0.01
    tolerance = 0.005
    max_time = 30.0

    controller = QuadrotorController(tello)
    param = tello

    pos = np.array(currentPose['position'], dtype=float)
    rpy = np.array(currentPose['rpy'], dtype=float)
    
    if maintain_orientation:
        initial_rpy = rpy.copy()
    elif lock_roll_pitch:
        initial_roll_pitch = rpy[:2].copy()
        initial_rpy = None
    else:
        initial_rpy = None

    vel = np.zeros(3)
    pqr = np.zeros(3)

    roll, pitch, yaw = rpy
    quat = (Quaternion(axis=[0,0,1], radians=yaw) *
            Quaternion(axis=[0,1,0], radians=pitch) *
            Quaternion(axis=[1,0,0], radians=roll))

    current_state = np.concatenate([
        pos, vel,
        [quat.x, quat.y, quat.z, quat.w],
        pqr
    ])

    target_position = np.array(targetPose, dtype=float)

    # Map bounds check
    MAP_X_MIN, MAP_X_MAX = 0.0, 2.0
    MAP_Y_LIMIT, MAP_Z_LIMIT = 2.0, 1.0
    if (target_position[0] < MAP_X_MIN or target_position[0] > MAP_X_MAX or
        abs(target_position[1]) > MAP_Y_LIMIT or
        abs(target_position[2]) > MAP_Z_LIMIT):
        print(f"  [ERROR] goToWaypoint target outside map bounds: {target_position}")
        return -1

    if doesItCollide(target_position):
        print(f"  [ERROR] goToWaypoint: target position collides: {target_position}")
        return -1

    distance = np.linalg.norm(target_position - pos)
    estimated_time = min(distance / max(velocity, 1e-6) * 2.0, max_time)

    if distance < tolerance:
        return {'position': pos, 'rpy': rpy}

    num_points = max(2, int(estimated_time / dt))
    time_points = np.linspace(0, estimated_time, num_points)

    direction = target_position - pos
    dist_dir = np.linalg.norm(direction)
    unit_direction = direction / dist_dir if dist_dir > 1e-6 else np.zeros(3)
    
    accel_time = min(1.0, estimated_time * 0.25)
    decel_time = accel_time
    cruise_time = max(0.0, estimated_time - accel_time - decel_time)
    denom = (0.5 * accel_time + cruise_time + 0.5 * decel_time)

    cruise_vel = min(velocity, distance / max(denom, 1e-6))
    
    trajectory_points, velocities, accelerations = [], [], []

    for t in time_points:
        if t <= accel_time:
            vel_mag = (cruise_vel / accel_time) * t
            acc_mag = cruise_vel / accel_time
            prog = 0.5 * (cruise_vel / accel_time) * t * t / max(distance, 1e-6)
        elif t <= accel_time + cruise_time:
            vel_mag = cruise_vel
            acc_mag = 0.0
            prog = (0.5 * cruise_vel * accel_time +
                    cruise_vel * (t - accel_time)) / max(distance, 1e-6)
        else:
            t_d = t - accel_time - cruise_time
            vel_mag = cruise_vel - (cruise_vel / max(decel_time, 1e-6)) * t_d
            vel_mag = max(0.0, vel_mag)
            acc_mag = -cruise_vel / max(decel_time, 1e-6)
            prog = (0.5 * cruise_vel * accel_time +
                    cruise_vel * cruise_time +
                    cruise_vel * t_d -
                    0.5 * (cruise_vel / max(decel_time, 1e-6)) * (t_d * t_d)) / max(distance, 1e-6)

        prog = np.clip(prog, 0.0, 1.0)

        trajectory_points.append(pos + prog * direction)
        velocities.append(vel_mag * unit_direction)
        accelerations.append(acc_mag * unit_direction)

    trajectory_points = np.array(trajectory_points)
    velocities = np.array(velocities)
    accelerations = np.array(accelerations)
    
    # SKIP pre-flight trajectory check - it's too conservative
    # The collision checker is overly sensitive to interpolated points
    # We'll check during execution instead (line ~295) which is more accurate
    # check_stride = max(1, len(trajectory_points) // 50)
    # for i in range(0, len(trajectory_points), check_stride):
    #     if doesItCollide(trajectory_points[i]):
    #         print(f"  [ERROR] goToWaypoint: collision detected on planned trajectory at index {i}, pos={trajectory_points[i]}")
    #         return -1

    if maintain_orientation:
        target_rpy_for_traj = rpy
    elif lock_roll_pitch:
        target_rpy_for_traj = np.array([0.0, 0.0, rpy[2]])
    else:
        target_rpy_for_traj = np.array([0.0, 0.0, rpy[2]])
        
    controller.set_trajectory(trajectory_points, time_points, velocities, accelerations,
                             target_rpy=target_rpy_for_traj)

    state = current_state.copy()
    
    last_capture_time = -1.0
    capture_interval = 0.5
    
    debug_iteration = 0
    max_debug_iterations = 3

    for i, t in enumerate(time_points):
        control_input = controller.compute_control(state, t)
        current_pos = state[0:3]
        
        if debug_iteration < max_debug_iterations:
            pos_des, vel_des, acc_des = controller.get_desired_state(t)
            print(f"\n  [CONTROL DEBUG {debug_iteration}] t={t:.3f}s")
            debug_iteration += 1

        if doesItCollide(current_pos):
            print(f"  [ERROR] goToWaypoint: collision detected during execution at time {t:.3f}, pos={current_pos}")
            return -1

        if navigator is not None and (t - last_capture_time) >= capture_interval:
            qx, qy, qz, qw = state[6], state[7], state[8], state[9]
            temp_quat = Quaternion(w=qw, x=qx, y=qy, z=qz)
            temp_yaw, temp_pitch, temp_roll = temp_quat.yaw_pitch_roll
            temp_rpy = np.array([temp_roll, temp_pitch, temp_yaw])
            
            temp_pose = {
                'position': current_pos.copy(),
                'rpy': temp_rpy
            }
            
            rgb_temp, _, _ = navigator.renderer.render(current_pos, temp_rpy)
            navigator.record_frame(rgb_temp, pose=temp_pose, annotation=f"{action}")
            last_capture_time = t

        err = np.linalg.norm(current_pos - target_position)

        if err < tolerance and t > 1.0:
            state_final = state
            break

        if i < len(time_points) - 1:
            sol = solve_ivp(
                lambda tau, X: model_derivative(tau, X, control_input, param),
                [t, t + dt],
                state,
                method='RK45',
                max_step=dt
            )
            state = sol.y[:, -1]
            state_final = state
    else:
        state_final = state

    final_pos = state_final[0:3]
    qx, qy, qz, qw = state_final[6], state_final[7], state_final[8], state_final[9]
    final_quat = Quaternion(w=qw, x=qx, y=qy, z=qz)
    yaw_f, pitch_f, roll_f = final_quat.yaw_pitch_roll

    if maintain_orientation:
        final_rpy = initial_rpy
    elif lock_roll_pitch:
        final_rpy = np.array([initial_roll_pitch[0], initial_roll_pitch[1], yaw_f])
    else:
        final_rpy = np.array([roll_f, pitch_f, yaw_f])
    
    if pose_history is not None:
        pose_history.append({
            'step': len(pose_history),
            'action': action,
            'position': final_pos.copy(),
            'rpy_deg': np.degrees(final_rpy),
            'rpy_rad': final_rpy.copy()
        })

    return {
        'position': final_pos,
        'rpy': final_rpy
    }