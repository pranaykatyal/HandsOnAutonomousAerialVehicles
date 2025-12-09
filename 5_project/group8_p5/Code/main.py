"""
Drone Racing Navigation System - FIXED VERSION
Sequential reactive window detection and navigation
With comprehensive collision avoidance
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
from window_detector import OpticalFlowExtractor, SimpleFlowDetector, ActiveScanner
import os


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def is_path_clear(start, end, num_checks=10):
    """
    Check if straight-line path between two points is collision-free
    
    Args:
        start: Starting position [x, y, z]
        end: Ending position [x, y, z]
        num_checks: Number of points to check along path
        
    Returns:
        is_clear: True if path is clear
        first_collision_idx: Index of first collision (-1 if clear)
    """
    for i in range(num_checks):
        t = i / (num_checks - 1) if num_checks > 1 else 0
        pos = start + t * (end - start)
        
        if doesItCollide(pos):
            return False, i
    
    return True, -1


class WindowNavigator:
    """
    Manages window detection and navigation
    """
    
    def __init__(self, renderer, device='cuda'):
        self.renderer = renderer
        self.device = device
        
        # Initialize optical flow detector
        raft_model_path = './RAFT/models/raft-things.pth'
        self.flow_extractor = OpticalFlowExtractor(raft_model_path, device=device)
        self.detector = SimpleFlowDetector(self.flow_extractor, device=device)
        self.scanner = ActiveScanner(scan_distance=0.15, num_waypoints=5)  # Increased scan distance
        
        # Store detected windows
        self.detected_windows = []
        self.window_count = 0
        
        # Video recording
        self.video_frames = []
        
        print("✓ Window Navigator initialized")
    
    def scan_for_window(self, current_pose):
        """
        Perform scanning motion to detect window
        """
        print("\n=== SCANNING FOR WINDOW ===")
        
        # Generate scanning waypoints
        scan_waypoints = self.scanner.generate_scan_trajectory(current_pose)
        
        print(f"  Generated {len(scan_waypoints)} scan waypoints")
        
        # Execute scanning motion and capture frames
        scan_frames = []
        scan_poses = []
        
        temp_pose = current_pose.copy()
        
        for i, waypoint in enumerate(scan_waypoints):
            # Navigate to waypoint
            result = goToWaypoint(temp_pose, waypoint, velocity=0.05)
            
            if result == -1:
                print(f"  Collision during scanning at waypoint {i}")
                return None, scan_frames
            
            temp_pose = result
            scan_poses.append(temp_pose)
            
            # Capture frame with camera orientation correction
            # CRITICAL: Camera might be upside down - apply 180° roll correction
            corrected_rpy = temp_pose['rpy'].copy()
            corrected_rpy[0] += np.pi  # Add 180° to roll (rotate around X-axis)
            
            rgb, _, _ = self.renderer.render(temp_pose['position'], corrected_rpy)
            
            # Also flip image if needed (in case renderer doesn't handle it)
            # Uncomment if camera orientation correction in render doesn't work:
            # rgb = cv2.flip(rgb, -1)  # Flip both horizontally and vertically (180° rotation)
            
            scan_frames.append(rgb)
            
            print(f"  Captured frame {i+1}/{len(scan_waypoints)}")
        
        # Detect window from frames
        print("\n  Detecting window from scanned frames...")
        mask, center_2d, confidence, debug_info = self.detector.detect_window(scan_frames)
        
        # Save visualization
        os.makedirs('./log', exist_ok=True)
        self.detector.visualize(debug_info, f'./log/window_detection_{self.window_count}.png')
        
        if center_2d is None or confidence < 0.015:  # Lowered from 0.05 to 0.015
            print(f"  ✗ No window detected (confidence: {confidence:.3f})")
            return None, scan_frames
        
        print(f"  ✓ Window detected at pixel {center_2d} (confidence: {confidence:.3f})")
        
        # Estimate 3D position of window
        mid_idx = len(scan_poses) // 2
        ref_pose = scan_poses[mid_idx]
        ref_pos = ref_pose['position']
        ref_rpy = ref_pose['rpy']
        
        # Adaptive depth estimation
        img_h, img_w = scan_frames[0].shape[:2]
        u_pix = center_2d[0]
        v_pix = center_2d[1]
        
        window_pixels = np.sum(mask > 0.5)
        img_pixels = img_h * img_w
        window_size_ratio = window_pixels / img_pixels
        
        # Conservative depth estimation (closer = safer)
        if window_size_ratio > 0.15:
            estimated_depth = 1.2
        elif window_size_ratio > 0.08:
            estimated_depth = 2.0
        elif window_size_ratio > 0.03:
            estimated_depth = 2.5
        else:
            estimated_depth = 3.0
        
        print(f"  Window size ratio: {window_size_ratio:.3f}, estimated depth: {estimated_depth:.1f}m")
        
        # Normalized image coordinates
        u_norm = (u_pix - img_w/2) / img_w
        v_norm = (v_pix - img_h/2) / img_h
        
        # Camera intrinsics
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
        
        # Clamp vertical range
        window_3d[2] = np.clip(window_3d[2], -1.5, 1.5)
        
        print(f"  Estimated window 3D position: [{window_3d[0]:.2f}, {window_3d[1]:.2f}, {window_3d[2]:.2f}]")
        
        # Validate and adjust window position
        if doesItCollide(window_3d):
            print(f"  ✗ Window position collides!")
            
            # Try multiple adjustments
            adjustments = [
                ('closer', 0.7, None),
                ('closer', 0.8, None),
                ('further', 1.1, None),
                ('further', 1.3, None),
                ('left', None, np.array([0, -0.3, 0])),
                ('right', None, np.array([0, 0.3, 0])),
                ('up', None, np.array([0, 0, -0.3])),
                ('down', None, np.array([0, 0, 0.3])),
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
    
    def navigate_to_window(self, current_pose, window_3d_pos, approach_distance=0.7):
        """
        Navigate towards and through window with smart collision avoidance
        """
        print(f"\n=== NAVIGATING TO WINDOW ===")
        print(f"  Current: {current_pose['position']}")
        print(f"  Target: {window_3d_pos}")
        
        direction = window_3d_pos - current_pose['position']
        distance = np.linalg.norm(direction)
        
        if distance < approach_distance:
            print("  Already at window")
            return current_pose
        
        direction_norm = direction / distance
        approach_point = window_3d_pos - direction_norm * approach_distance
        
        # Check if approach point is accessible
        if doesItCollide(approach_point):
            print(f"  ✗ Approach point collides!")
            
            for new_dist in [1.0, 0.5, 1.5]:
                test_point = window_3d_pos - direction_norm * new_dist
                if not doesItCollide(test_point):
                    approach_point = test_point
                    approach_distance = new_dist
                    print(f"  ✓ Adjusted approach distance to {new_dist:.2f}m")
                    break
            else:
                print(f"  ✗ Cannot find collision-free approach")
                return -1
        
        # Check if direct path is clear
        path_clear, first_collision = is_path_clear(current_pose['position'], approach_point, num_checks=20)
        
        if path_clear:
            print(f"  ✓ Direct path is clear")
            waypoints = [approach_point]
        else:
            print(f"  ✗ Direct path blocked at check {first_collision}/20")
            print(f"  Searching for alternate path...")
            
            # Try systematic alternate paths
            waypoints = self.find_alternate_path(current_pose['position'], approach_point)
            
            if waypoints is None:
                print(f"  ✗ Cannot find collision-free path")
                return -1
        
        # Navigate through waypoints
        temp_pose = current_pose
        
        for i, waypoint in enumerate(waypoints):
            print(f"  Navigating to waypoint {i+1}/{len(waypoints)}: [{waypoint[0]:.2f}, {waypoint[1]:.2f}, {waypoint[2]:.2f}]")
            
            result = goToWaypoint(temp_pose, waypoint, velocity=0.08)
            
            if result == -1:
                print(f"  ✗ Collision during navigation")
                return -1
            
            temp_pose = result
            print(f"  ✓ Reached waypoint {i+1}/{len(waypoints)}")
        
        print(f"  ✓ Reached approach point")
        
        # Navigate through window
        through_point = window_3d_pos + direction_norm * 0.3
        
        if not doesItCollide(through_point):
            result = goToWaypoint(temp_pose, through_point, velocity=0.06)
            
            if result != -1:
                print(f"  ✓ Passed through window")
                return result
        
        print(f"  Stopping at approach point (through point blocked)")
        return temp_pose
    
    def find_alternate_path(self, start, goal, max_waypoints=20):
        """
        Find collision-free path using greedy exploration
        """
        direction = goal - start
        distance = np.linalg.norm(direction)
        direction_norm = direction / distance
        
        # Strategy 1: Offset paths with SMALLER steps
        perpendicular = np.array([-direction_norm[1], direction_norm[0], 0])
        perpendicular = perpendicular / (np.linalg.norm(perpendicular) + 1e-6)
        
        vertical = np.array([0, 0, -1])
        
        offsets_to_try = [
            (perpendicular, 0.3, 'right-close'),
            (-perpendicular, 0.3, 'left-close'),
            (perpendicular, 0.5, 'right-med'),
            (-perpendicular, 0.5, 'left-med'),
            (perpendicular, 0.7, 'right-far'),
            (-perpendicular, 0.7, 'left-far'),
            (vertical, 0.2, 'up-slight'),
            (-vertical, 0.2, 'down-slight'),
            (vertical, 0.4, 'up-med'),
            (-vertical, 0.4, 'down-med'),
        ]
        
        # Use smaller segments for better granularity
        num_segments = max(5, min(int(distance / 0.3), max_waypoints))  # One every 0.3m
        
        for offset_dir, offset_mag, offset_name in offsets_to_try:
            waypoints = []
            all_clear = True
            
            for i in range(1, num_segments + 1):
                t = i / num_segments
                base = start + t * (goal - start)
                # Smooth offset (sine wave)
                offset_scale = np.sin(t * np.pi)
                pos = base + offset_dir * offset_mag * offset_scale
                
                # Check this point AND path to it from previous
                if doesItCollide(pos):
                    all_clear = False
                    break
                
                # Also check path from previous waypoint
                if len(waypoints) > 0:
                    prev = waypoints[-1]
                    mid_check = (prev + pos) / 2
                    if doesItCollide(mid_check):
                        all_clear = False
                        break
                
                waypoints.append(pos)
            
            if all_clear:
                print(f"    ✓ Found {offset_name} path with {len(waypoints)} waypoints")
                return waypoints
        
        # Strategy 2: Cautious step-by-step exploration
        print(f"    Trying step-by-step exploration...")
        current = start.copy()
        waypoints = []
        step_size = 0.15  # Smaller steps
        
        for step in range(max_waypoints * 2):  # Allow more steps
            # Try moving toward goal
            next_pos = current + direction_norm * step_size
            
            if doesItCollide(next_pos):
                # Try all lateral/vertical directions
                found_move = False
                for lateral in [perpendicular, -perpendicular, vertical, -vertical,
                               perpendicular + vertical, -perpendicular + vertical]:
                    test_pos = current + lateral * (step_size * 0.7)
                    if not doesItCollide(test_pos):
                        next_pos = test_pos
                        found_move = True
                        break
                
                if not found_move:
                    break
            
            waypoints.append(next_pos)
            current = next_pos
            
            # Check if close to goal
            if np.linalg.norm(current - goal) < step_size * 3:
                # Try to reach goal directly
                if not doesItCollide(goal):
                    is_clear, _ = is_path_clear(current, goal, num_checks=5)
                    if is_clear:
                        waypoints.append(goal)
                        print(f"    ✓ Exploration path with {len(waypoints)} waypoints")
                        return waypoints
                break
        
        if len(waypoints) >= 3:
            print(f"    ⚠ Partial path with {len(waypoints)} waypoints")
            return waypoints
        
        return None
    
    def record_frame(self, rgb_frame, mask=None):
        """Record frame for video"""
        if mask is not None:
            overlay = rgb_frame.copy()
            mask_color = np.zeros_like(overlay)
            mask_color[mask > 0.5] = [0, 255, 0]
            frame = cv2.addWeighted(overlay, 0.7, mask_color, 0.3, 0)
        else:
            frame = rgb_frame
        
        self.video_frames.append(frame)
    
    def save_video(self, output_path='./log/navigation_video.mp4', fps=30):
        """Save recorded frames as video"""
        if not self.video_frames:
            print("No frames to save")
            return
        
        h, w = self.video_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        for frame in self.video_frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        out.release()
        print(f"✓ Saved video to {output_path} ({len(self.video_frames)} frames)")


def goToWaypoint(currentPose, targetPose, velocity=0.1):
    """Navigate quadrotor to a target waypoint"""
    dt = 0.01
    tolerance = 0.1
    max_time = 30.0

    controller = QuadrotorController(tello)
    param = tello

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
    
    # CRITICAL: Pre-validate trajectory before attempting navigation
    # Check every 10th point to avoid excessive checking
    check_stride = max(1, len(trajectory_points) // 50)  # Check ~50 points
    for i in range(0, len(trajectory_points), check_stride):
        if doesItCollide(trajectory_points[i]):
            print(f'  ✗ Trajectory collision at point {i}/{len(trajectory_points)} [{trajectory_points[i][0]:.2f}, {trajectory_points[i][1]:.2f}, {trajectory_points[i][2]:.2f}]')
            return -1

    controller.set_trajectory(trajectory_points, time_points, velocities, accelerations)

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

    final_rpy = np.array([roll_f, pitch_f, yaw_f])

    return {
        'position': final_pos,
        'rpy': final_rpy
    }


def main(renderer):
    os.makedirs('./log', exist_ok=True)
    
    print("\n" + "="*60)
    print("DRONE RACING")
    print("="*60 + "\n")

    navigator = WindowNavigator(renderer, device='cuda')
    
    currentPose = {
        'position': np.array([0.2, -0.2, 0.0]),  # Working position from P4
        'rpy': np.radians([0.0, 0.0, 0.0])
    }
    
    if doesItCollide(currentPose['position']):
        print('✗ Starting position collides!')
        return -1
    
    # Capture initial frame with camera orientation correction
    corrected_rpy_initial = currentPose['rpy'].copy()
    corrected_rpy_initial[0] += np.pi  # Add 180° to roll
    rgb, _, _ = renderer.render(currentPose['position'], corrected_rpy_initial)
    navigator.record_frame(rgb)
    
    print("\n" + "="*60)
    print("PHASE 1: FORWARD NAVIGATION")
    print("="*60)
    
    print(f"\nStarting from known good position: {currentPose['position']}")
    
    detected_windows = []
    max_windows = 10
    no_detection_count = 0
    max_no_detection = 3  # Increased tolerance
    
    for window_num in range(max_windows):
        print(f"\n--- Window {window_num + 1} ---")
        
        window_3d_pos, scan_frames = navigator.scan_for_window(currentPose)
        
        if window_3d_pos is None:
            no_detection_count += 1
            print(f"  No window detected ({no_detection_count}/{max_no_detection})")
            
            if no_detection_count >= max_no_detection:
                print("\n  No more windows detected. Proceeding to return phase.")
                break
            
            # Try moving forward cautiously
            forward_test = currentPose['position'] + np.array([0.5, 0.0, 0.0])
            if not doesItCollide(forward_test):
                result = goToWaypoint(currentPose, forward_test, velocity=0.08)
                if result != -1:
                    currentPose = result
                    continue
            
            # Try moving laterally
            lateral_test = currentPose['position'] + np.array([0.0, 0.3, 0.0])
            if not doesItCollide(lateral_test):
                result = goToWaypoint(currentPose, lateral_test, velocity=0.08)
                if result != -1:
                    currentPose = result
                    continue
            
            break
        
        no_detection_count = 0
        detected_windows.append(window_3d_pos)
        navigator.window_count += 1
        
        result = navigator.navigate_to_window(currentPose, window_3d_pos)
        
        if result == -1:
            print(f"  ✗ Failed to navigate through window {window_num + 1}")
            print(f"  Environment appears too cluttered for navigation")
            print(f"  Switching to DETECTION-ONLY mode...")
            
            # Enter detection-only mode
            print("\n  === DETECTION-ONLY MODE ===")
            print("  Continuing to scan and detect windows without navigation")
            
            for detect_only_num in range(window_num + 2, max_windows + 1):
                print(f"\n  --- Detection {detect_only_num} ---")
                
                # Just scan without moving between scans
                window_3d_pos_detect, _ = navigator.scan_for_window(currentPose)
                
                if window_3d_pos_detect is not None:
                    detected_windows.append(window_3d_pos_detect)
                    navigator.window_count += 1
                    print(f"  ✓ Window {navigator.window_count} detected (detection-only)")
                else:
                    no_detection_count += 1
                    if no_detection_count >= max_no_detection:
                        break
            
            print(f"\n  Detection-only mode complete. Found {len(detected_windows)} total windows.")
            break
        
        currentPose = result
        print(f"  ✓ Successfully navigated through window {window_num + 1}")
        
        # Capture frame with camera correction
        corrected_rpy_nav = currentPose['rpy'].copy()
        corrected_rpy_nav[0] += np.pi
        rgb, _, _ = renderer.render(currentPose['position'], corrected_rpy_nav)
        navigator.record_frame(rgb)
    
    print("\n" + "="*60)
    print("PHASE 2: RETURN NAVIGATION")
    print("="*60)
    
    if len(detected_windows) > 0:
        print(f"  Returning through {len(detected_windows)} windows in reverse")
        
        for i, window_pos in enumerate(reversed(detected_windows)):
            print(f"\n--- Returning through window {i + 1}/{len(detected_windows)} ---")
            
            result = navigator.navigate_to_window(currentPose, window_pos)
            
            if result == -1:
                print(f"  ✗ Failed during return, continuing...")
                continue
            
            currentPose = result
            
            # Capture frame with camera correction
            corrected_rpy_return = currentPose['rpy'].copy()
            corrected_rpy_return[0] += np.pi
            rgb, _, _ = renderer.render(currentPose['position'], corrected_rpy_return)
            navigator.record_frame(rgb)
        
        print("\n--- Returning to start position ---")
        start_position = np.array([0.0, 0.0, 0.0])
        
        if not doesItCollide(start_position):
            result = goToWaypoint(currentPose, start_position, velocity=0.1)
            
            if result != -1:
                currentPose = result
                print("  ✓ Returned to start position!")
    
    navigator.save_video()
    
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