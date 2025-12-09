"""
Drone Racing Navigation System - FIXED VERSION v3
- Pose history tracking
- Better component selection (prefers CENTER over edges)
- Proper renderer usage
- Collision parameter fixes
"""

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

# NOTE: Using DEFAULT collision parameters until we calibrate the scale
# The Gaussian splat coordinate system may not be 1:1 with meters
# Default: drone_radius=0.001, collision_threshold=0.01
# 
# Run test_collision_scale.py to find the actual scale!
# After calibration, you can create a wrapper with appropriate parameters

from window_detector import OpticalFlowExtractor, SimpleFlowDetector, ActiveScanner
import os
import json


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def is_path_clear(start, end, num_checks=10):
    """Check if straight-line path is collision-free"""
    for i in range(num_checks):
        t = i / (num_checks - 1) if num_checks > 1 else 0
        pos = start + t * (end - start)
        
        if doesItCollide(pos):
            return False, i
    
    return True, -1


class WindowNavigator:
    """Manages window detection and navigation"""
    
    def __init__(self, renderer, device='cuda'):
        self.renderer = renderer
        self.device = device
        
        # Initialize optical flow detector
        raft_model_path = './RAFT/models/raft-things.pth'
        self.flow_extractor = OpticalFlowExtractor(raft_model_path, device=device)
        self.detector = SimpleFlowDetector(self.flow_extractor, device=device)
        
        # CRITICAL: Scan distance must be >> goToWaypoint tolerance (0.1m)
        # Otherwise drone doesn't actually move and we get zero parallax!
        # Each waypoint step = scan_distance / (num_waypoints - 1)
        # With 5 waypoints: step = 0.6 / 4 = 0.15m per step >> 0.1m tolerance ✓
        self.scanner = ActiveScanner(scan_distance=0.6, num_waypoints=5)
        
        
        self.detected_windows = []
        self.window_count = 0
        self.video_frames = []
        
        print("✓ Window Navigator initialized")
    
    def scan_for_window(self, current_pose, pose_history):
        """Perform scanning motion to detect window"""
        print("\n=== SCANNING FOR WINDOW ===")
        
        scan_waypoints = self.scanner.generate_scan_trajectory(current_pose)
        print(f"  Generated {len(scan_waypoints)} scan waypoints")
        
        # DEBUG: Print waypoint distances
        print(f"  Scan waypoint distances from current:")
        for i, wp in enumerate(scan_waypoints):
            dist = np.linalg.norm(wp - current_pose['position'])
            print(f"    WP{i}: distance = {dist:.4f}m")
        
        scan_frames = []
        scan_poses = []
        temp_pose = current_pose.copy()
        
        for i, waypoint in enumerate(scan_waypoints):
            print(f"\n  Navigating to scan waypoint {i+1}/{len(scan_waypoints)}")
            print(f"    Current: {temp_pose['position']}")
            print(f"    Target: {waypoint}")
            
            result = goToWaypoint(temp_pose, waypoint, velocity=0.05, pose_history=pose_history, 
                                 action=f'SCAN_WP{i+1}', lock_roll_pitch=True)
            
            if result == -1:
                print(f"  Collision during scanning at waypoint {i}")
                return None, scan_frames
            
            temp_pose = result
            scan_poses.append(temp_pose)
            
            print(f"    Reached: {temp_pose['position']}")
            print(f"    Actual movement: {np.linalg.norm(temp_pose['position'] - current_pose['position']):.4f}m")
            #                                              ^^^^^^^^^^^^^^^^ Lock roll/pitch, allow yaw
            
            if result == -1:
                print(f"  Collision during scanning at waypoint {i}")
                return None, scan_frames
            
            temp_pose = result
            scan_poses.append(temp_pose)
            
            # Camera orientation: NO correction needed if camera is right-side up
            rgb, _, _ = self.renderer.render(temp_pose['position'], temp_pose['rpy'])
            
            scan_frames.append(rgb)
            self.record_frame(rgb, frame_id=len(self.video_frames))
            
            print(f"  Captured frame {i+1}/{len(scan_waypoints)}")
        
        # Detect window
        print("\n  Detecting window from scanned frames...")
        mask, center_2d, confidence, debug_info = self.detector.detect_window(scan_frames)
        
        # Save visualization
        os.makedirs('./log', exist_ok=True)
        self.detector.visualize(debug_info, f'./log/window_detection_{self.window_count}.png')
        
        if center_2d is None or confidence < 0.015:
            print(f"  ✗ No window detected (confidence: {confidence:.3f})")
            return None, scan_frames
        
        print(f"  ✓ Window detected at pixel {center_2d} (confidence: {confidence:.3f})")
        
        # Estimate 3D position
        mid_idx = len(scan_poses) // 2
        ref_pose = scan_poses[mid_idx]
        ref_pos = ref_pose['position']
        ref_rpy = ref_pose['rpy']
        
        img_h, img_w = scan_frames[0].shape[:2]
        u_pix = center_2d[0]
        v_pix = center_2d[1]
        
        window_pixels = np.sum(mask > 0.5)
        img_pixels = img_h * img_w
        window_size_ratio = window_pixels / img_pixels
        
        # Depth estimation
        if window_size_ratio > 0.15:
            estimated_depth = 1.2
        elif window_size_ratio > 0.08:
            estimated_depth = 2.0
        elif window_size_ratio > 0.03:
            estimated_depth = 2.5
        else:
            estimated_depth = 3.0
        
        print(f"  Window size ratio: {window_size_ratio:.3f}, estimated depth: {estimated_depth:.1f}m")
        
        # Normalized coordinates
        u_norm = (u_pix - img_w/2) / img_w
        v_norm = (v_pix - img_h/2) / img_h
        
        # Camera parameters
        fov_rad = 1.3089969389957472
        aspect_ratio = img_w / img_h
        
        # Camera frame offsets
        offset_x_cam = estimated_depth
        offset_y_cam = estimated_depth * np.tan(fov_rad/2) * aspect_ratio * u_norm * 2
        offset_z_cam = estimated_depth * np.tan(fov_rad/2) * v_norm * 2
        
        # Transform to NED
        roll, pitch, yaw = ref_rpy
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        offset_north = offset_x_cam * cos_yaw - offset_y_cam * sin_yaw
        offset_east = offset_x_cam * sin_yaw + offset_y_cam * cos_yaw
        offset_down = offset_z_cam
        
        window_3d = np.array([
            ref_pos[0] + offset_north,
            ref_pos[1] + offset_east,
            ref_pos[2] + offset_down
        ])
        
        window_3d[2] = np.clip(window_3d[2], -1.5, 1.5)
        
        print(f"  Estimated window 3D position: [{window_3d[0]:.2f}, {window_3d[1]:.2f}, {window_3d[2]:.2f}]")
        
        # Validate position
        if doesItCollide(window_3d):
            print(f"  ✗ Window position collides, trying adjustments...")
            
            adjustments = [
                ('closer', 0.7, None),
                ('further', 1.3, None),
                ('left', None, np.array([0, -0.3, 0])),
                ('right', None, np.array([0, 0.3, 0])),
            ]
            
            for adj_name, scale, offset in adjustments:
                if scale is not None:
                    adjusted = ref_pos + (window_3d - ref_pos) * scale
                else:
                    adjusted = window_3d + offset
                
                if not doesItCollide(adjusted):
                    print(f"  ✓ Adjusted ({adj_name}): [{adjusted[0]:.2f}, {adjusted[1]:.2f}, {adjusted[2]:.2f}]")
                    window_3d = adjusted
                    break
            else:
                print(f"  ✗ Could not find collision-free position")
                return None, scan_frames
        
        return window_3d, scan_frames
    
    def navigate_to_window(self, current_pose, window_3d_pos, pose_history, approach_distance=0.7):
        """Navigate to window with yaw alignment"""
        print(f"\n=== NAVIGATING TO WINDOW ===")
        print(f"  Current: {current_pose['position']}")
        print(f"  Target: {window_3d_pos}")
        
        direction = window_3d_pos - current_pose['position']
        distance = np.linalg.norm(direction)
        
        if distance < approach_distance:
            print("  Already at window")
            return current_pose
        
        # CRITICAL: Calculate desired yaw to face the window
        desired_yaw = np.arctan2(direction[1], direction[0])  # atan2(East, North) in NED
        current_yaw = current_pose['rpy'][2]
        
        yaw_error = wrap_angle(desired_yaw - current_yaw)
        
        print(f"  Current yaw: {np.degrees(current_yaw):.1f}°")
        print(f"  Desired yaw (toward window): {np.degrees(desired_yaw):.1f}°")
        print(f"  Yaw error: {np.degrees(yaw_error):.1f}°")
        
        # If yaw error > 10°, first rotate to face window
        if abs(yaw_error) > np.radians(10):
            print(f"  ✓ Rotating to face window...")
            aligned_pose = current_pose.copy()
            aligned_pose['rpy'] = np.array([0.0, 0.0, desired_yaw])
            current_pose = aligned_pose
            
            # Record yaw alignment in pose history
            if pose_history is not None:
                pose_history.append({
                    'step': len(pose_history),
                    'action': 'YAW_ALIGN',
                    'position': current_pose['position'].copy(),
                    'rpy_deg': np.degrees(current_pose['rpy']),
                    'rpy_rad': current_pose['rpy'].copy()
                })
        
        direction_norm = direction / distance
        approach_point = window_3d_pos - direction_norm * approach_distance
        
        if doesItCollide(approach_point):
            print(f"  ✗ Approach point collides!")
            return -1
        
        # Check path
        path_clear, first_collision = is_path_clear(current_pose['position'], approach_point, num_checks=20)
        
        if path_clear:
            print(f"  ✓ Direct path is clear")
            waypoints = [approach_point]
        else:
            print(f"  ✗ Direct path blocked at check {first_collision}/20")
            return -1
        
        # Navigate
        temp_pose = current_pose
        
        for i, waypoint in enumerate(waypoints):
            print(f"  Navigating to waypoint {i+1}/{len(waypoints)}: [{waypoint[0]:.2f}, {waypoint[1]:.2f}, {waypoint[2]:.2f}]")
            
            result = goToWaypoint(temp_pose, waypoint, velocity=0.08, pose_history=pose_history,
                                 action=f'NAV_WP{i+1}')
            
            if result == -1:
                print(f"  ✗ Collision during navigation")
                print(f"  Last valid pose: pos={temp_pose['position']}, rpy_deg={np.degrees(temp_pose['rpy'])}")
                return -1
            
            temp_pose = result
            print(f"  ✓ Reached waypoint {i+1}/{len(waypoints)}")
            print(f"    Current pose: pos=[{temp_pose['position'][0]:.3f}, {temp_pose['position'][1]:.3f}, {temp_pose['position'][2]:.3f}], "
                  f"rpy=[{np.degrees(temp_pose['rpy'][0]):.1f}, {np.degrees(temp_pose['rpy'][1]):.1f}, {np.degrees(temp_pose['rpy'][2]):.1f}]°")
        
        print(f"  ✓ Reached approach point")
        return temp_pose
    
    def record_frame(self, rgb_frame, mask=None, frame_id=None):
        """Save frame to disk"""
        if len(self.video_frames) == 0:
            import glob
            for f in glob.glob('./log/frames/*.png'):
                try:
                    os.remove(f)
                except:
                    pass
        
        if mask is not None:
            overlay = rgb_frame.copy()
            mask_color = np.zeros_like(overlay)
            mask_color[mask > 0.5] = [0, 255, 0]
            frame = cv2.addWeighted(overlay, 0.7, mask_color, 0.3, 0)
        else:
            frame = rgb_frame
        
        self.video_frames.append(frame)
        
        if frame_id is None:
            frame_id = len(self.video_frames) - 1
        
        frame_path = f'./log/frames/frame_{frame_id:04d}.png'
        os.makedirs('./log/frames', exist_ok=True)
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    def save_frames_summary(self):
        """Create summary"""
        num_frames = len(self.video_frames)
        if num_frames == 0:
            print("No frames recorded")
            return
        
        print(f"\n✓ Saved {num_frames} frames to ./log/frames/")
        print(f"  frame_0000.png through frame_{num_frames-1:04d}.png")


def goToWaypoint(currentPose, targetPose, velocity=0.1, pose_history=None, action='NAV', maintain_orientation=False, lock_roll_pitch=False):
    """
    Navigate to waypoint with pose tracking
    
    Args:
        maintain_orientation: If True, keep initial roll/pitch/yaw constant
        lock_roll_pitch: If True, only lock roll/pitch but allow yaw changes
    """
    dt = 0.01
    tolerance = 0.1
    max_time = 30.0

    controller = QuadrotorController(tello)
    param = tello

    pos = np.array(currentPose['position'], dtype=float)
    rpy = np.array(currentPose['rpy'], dtype=float)
    
    # Store initial orientation based on locking mode
    if maintain_orientation:
        initial_rpy = rpy.copy()  # Lock all: roll, pitch, yaw
    elif lock_roll_pitch:
        initial_roll_pitch = rpy[:2].copy()  # Only lock roll and pitch
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

    if doesItCollide(target_position):
        print('  ✗ Target position collides!')
        return -1

    distance = np.linalg.norm(target_position - pos)
    estimated_time = min(distance / max(velocity,1e-6) * 2.0, max_time)

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
    
    # Pre-validate trajectory
    check_stride = max(1, len(trajectory_points) // 50)
    for i in range(0, len(trajectory_points), check_stride):
        if doesItCollide(trajectory_points[i]):
            print(f'  ✗ Trajectory collision at point {i}/{len(trajectory_points)} [{trajectory_points[i][0]:.2f}, {trajectory_points[i][1]:.2f}, {trajectory_points[i][2]:.2f}]')
            return -1

    # CRITICAL: If maintain_orientation, use initial RPY for setpoint
    # This prevents unwanted flipping during scanning
    target_rpy_for_traj = initial_rpy if maintain_orientation else rpy
    controller.set_trajectory(trajectory_points, time_points, velocities, accelerations, 
                             target_rpy=target_rpy_for_traj)

    state = current_state.copy()

    for i, t in enumerate(time_points):
        control_input = controller.compute_control(state, t)
        current_pos = state[0:3]

        if doesItCollide(current_pos):
            print('  ✗ Collision during execution!')
            return -1

        err = np.linalg.norm(current_pos - target_position)

        if err < tolerance and t > 1.0:
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

    # Apply orientation constraints
    if maintain_orientation:
        # Lock all: roll, pitch, yaw
        final_rpy = initial_rpy
    elif lock_roll_pitch:
        # Lock only roll and pitch, allow yaw to change
        final_rpy = np.array([initial_roll_pitch[0], initial_roll_pitch[1], yaw_f])
    else:
        # Allow all to change
        final_rpy = np.array([roll_f, pitch_f, yaw_f])
    
    # Record pose
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


def main(renderer):
    os.makedirs('./log', exist_ok=True)
    import glob
    os.makedirs('./log/frames', exist_ok=True)
    for f in glob.glob('./log/*.png'):
        try:
            os.remove(f)
        except:
            pass
    for f in glob.glob('./log/frames/*.png'):
        try:
            os.remove(f)
        except:
            pass
    
    # POSE HISTORY TRACKING
    pose_history = []
    
    print("\n" + "="*60)
    print("DRONE RACING")
    print("="*60 + "\n")

    navigator = WindowNavigator(renderer, device='cuda')
    
    currentPose = {
        'position': np.array([0.0, 0.0, 0.0]),  # Origin - see first window
        'rpy': np.radians([0.0, 0.0, 0.0])
    }
    
    # Record initial pose
    pose_history.append({
        'step': 0,
        'action': 'INITIAL',
        'position': currentPose['position'].copy(),
        'rpy_deg': np.degrees(currentPose['rpy']),
        'rpy_rad': currentPose['rpy'].copy()
    })
    
    print(f"Initial pose:")
    print(f"  Position: [{currentPose['position'][0]:.3f}, {currentPose['position'][1]:.3f}, {currentPose['position'][2]:.3f}]")
    print(f"  RPY (deg): [{np.degrees(currentPose['rpy'][0]):.1f}, {np.degrees(currentPose['rpy'][1]):.1f}, {np.degrees(currentPose['rpy'][2]):.1f}]")
    
    if doesItCollide(currentPose['position']):
        print('✗ Starting position collides!')
        return -1
    
    # Capture initial frame
    rgb, _, _ = renderer.render(currentPose['position'], currentPose['rpy'])
    navigator.record_frame(rgb)
    
    print("\n" + "="*60)
    print("PHASE 1: FORWARD NAVIGATION")
    print("="*60)
    
    detected_windows = []
    max_windows = 3  # Try fewer windows
    
    for window_num in range(max_windows):
        print(f"\n--- Window {window_num + 1} ---")
        
        window_3d_pos, scan_frames = navigator.scan_for_window(currentPose, pose_history)
        
        if window_3d_pos is None:
            print(f"  No window detected")
            break
        
        detected_windows.append(window_3d_pos)
        navigator.window_count += 1
        
        result = navigator.navigate_to_window(currentPose, window_3d_pos, pose_history)
        
        if result == -1:
            print(f"  ✗ Failed to navigate through window {window_num + 1}")
            break
        
        currentPose = result
        print(f"  ✓ Successfully navigated through window {window_num + 1}")
        
        # Capture frame
        rgb, _, _ = renderer.render(currentPose['position'], currentPose['rpy'])
        navigator.record_frame(rgb)
    
    # Save pose history
    print("\n" + "="*60)
    print("SAVING POSE HISTORY")
    print("="*60)
    
    with open('./log/pose_history.json', 'w') as f:
        json.dump(pose_history, f, indent=2, default=str)
    
    print(f"✓ Saved {len(pose_history)} poses to ./log/pose_history.json")
    
    # Print summary
    print("\nPose Summary:")
    for i, p in enumerate(pose_history):
        print(f"  {i}: {p['action']:12s} pos=[{p['position'][0]:6.2f}, {p['position'][1]:6.2f}, {p['position'][2]:6.2f}] "
              f"rpy=[{p['rpy_deg'][0]:6.1f}, {p['rpy_deg'][1]:6.1f}, {p['rpy_deg'][2]:6.1f}]°")
    
    navigator.save_frames_summary()
    
    print("\n" + "="*60)
    print("NAVIGATION COMPLETE")
    print(f"  Windows navigated: {len(detected_windows)}")
    print(f"  Final position: {currentPose['position']}")
    print("="*60 + "\n")


if __name__ == "__main__":
    config_path = "../data/P5_colmap_splat/P5_colmap/splatfacto/2025-11-17_130359/config.yml"
    json_path = "../data/render_settings/render_settings.json"

    renderer = SplatRenderer(config_path, json_path)
    main(renderer)