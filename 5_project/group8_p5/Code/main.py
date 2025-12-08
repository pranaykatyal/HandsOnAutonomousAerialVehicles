from splat_render import SplatRenderer
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pyquaternion import Quaternion
from control import QuadrotorController
from quad_dynamics import model_derivative
import tello
from collisionChecker import doesItCollide

def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

################################################
#### Navigation Function ########################
################################################
def goToWaypoint(currentPose, targetPose, velocity=0.1):
    """
    Navigate quadrotor to a target waypoint
    """


    dt = 0.01
    tolerance = 0.1
    max_time = 30.0

    controller = QuadrotorController(tello)
    param = tello

    # -----------------------------------------------
    # Extract true starting pose
    # -----------------------------------------------
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

    target_position = np.array(targetPose, dtype=float)

    # check for collision
    if doesItCollide(target_position):
        print('Your robot hit the obstacle !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!.')
        return -1

    distance = np.linalg.norm(target_position - pos)
    estimated_time = min(distance / max(velocity,1e-6) * 2.0, max_time)

    print(f"  Navigating (noisy start): {pos} → {target_position}")
    print(f"  Distance: {distance:.2f} m, Est. time: {estimated_time:.1f}s")

    if distance < tolerance:
        print("  Already at target.")
        return {'position': pos, 'rpy': rpy}

    num_points = max(2, int(estimated_time / dt))
    time_points = np.linspace(0, estimated_time, num_points)

    direction = target_position - pos
    dist_dir = np.linalg.norm(direction)
    unit_direction = direction / dist_dir if dist_dir > 1e-6 else np.zeros(3)

    accel_time = min(1.0, estimated_time * 0.25)
    decel_time = accel_time
    cruise_time = max(0.0, estimated_time - accel_time - decel_time)
    denom = (0.5*accel_time + cruise_time + 0.5*decel_time)

    cruise_vel = min(velocity, distance / max(denom, 1e-6))

    trajectory_points, velocities, accelerations = [], [], []

    for t in time_points:
        if t <= accel_time:
            vel_mag = (cruise_vel/accel_time)*t
            acc_mag = cruise_vel/accel_time
            prog = 0.5*(cruise_vel/accel_time)*t*t / max(distance,1e-6)
        elif t <= accel_time + cruise_time:
            vel_mag = cruise_vel
            acc_mag = 0.0
            prog = (0.5*cruise_vel*accel_time +
                    cruise_vel*(t-accel_time)) / max(distance,1e-6)
        else:
            t_d = t - accel_time - cruise_time
            vel_mag = cruise_vel - (cruise_vel/max(decel_time,1e-6))*t_d
            vel_mag = max(0.0, vel_mag)
            acc_mag = -cruise_vel/max(decel_time,1e-6)
            prog = (0.5*cruise_vel*accel_time +
                    cruise_vel*cruise_time +
                    cruise_vel*t_d -
                    0.5*(cruise_vel/max(decel_time,1e-6))*(t_d*t_d)) / max(distance,1e-6)

        prog = np.clip(prog, 0.0, 1.0)

        trajectory_points.append(pos + prog * direction)
        velocities.append(vel_mag * unit_direction)
        accelerations.append(acc_mag * unit_direction)

    trajectory_points = np.array(trajectory_points)
    velocities = np.array(velocities)
    accelerations = np.array(accelerations)

    controller.set_trajectory(trajectory_points, time_points, velocities, accelerations)

    # Simulate dynamics
    state = current_state.copy()

    for i, t in enumerate(time_points):

        control_input = controller.compute_control(state, t)

        current_pos = state[0:3]

        # check collision here
        if doesItCollide(current_pos):
            print('Your robot collided !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            return -1

        err = np.linalg.norm(current_pos - target_position)

        if err < tolerance and t > 1.0:
            print(f"  ✓ Reached at t={t:.2f}s, err={err:.3f} m")
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

    return {
        'position': final_pos,
        'rpy': final_rpy
    }


################################################
#### Main Function ##############################
################################################
def main(renderer):
    # Create log directory if it doesn't exist
    import os
    os.makedirs('./log', exist_ok=True)

    # SET THIS TRUE AFTER RETURNING TO STARTING POINT
    raceFinished = False

    # MEASURING RACE TIME
    timeCounter = 0
    
    # Initialize pose - Position: x, y, z in meters | Orientation: roll, pitch, yaw in radians
    currentPose = {
        'position': np.array([0.0, 0.0, 0.0]),  # NED origin
        'rpy': np.radians([0.0, 0.0, 0.0])      # Orientation origin
    }
    
    ##### Collision checker ###########
    if doesItCollide(currentPose['position']):
        print('Your robot hit the obstacle !!!!!!!!!!!!!!!!!!!!!!!!!')
        return -1


    # renderer
    color_image, _, _ = renderer.render(
        currentPose['position'], 
        currentPose['rpy'])
    

    while not raceFinished:
        timeCounter += 1
        #####################################################
        ### Add your navigation code below ####
        ### ...
        ### ...
        ### set raceFinished = True after returning to start
        #####################################################
    

if __name__ == "__main__":
    config_path = "../data/P5_colmap_splat/P5_colmap/splatfacto/2025-11-17_130359/config.yml"
    json_path = "../data/render_settings/render_settings.json"

    renderer = SplatRenderer(config_path, json_path)
    main(renderer)
