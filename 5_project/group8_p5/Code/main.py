"""
Drone Racing Navigation System - WITH PnP INTEGRATION
✅ PnP-based window pose estimation
✅ Yaw changes use goToWaypoint (not instant)
✅ Coordinate transformation fixed
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
from window_pnp import WindowPnPEstimator, get_camera_matrix_from_fov
import os
import json


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class WindowNavigator:
    """Manages window detection, alignment, and navigation with PnP pose estimation"""
    
    def __init__(self, renderer, device='cuda'):
        self.renderer = renderer
        self.device = device
        
        # Initialize optical flow detector
        raft_model_path = './RAFT/models/raft-things.pth'
        self.flow_extractor = OpticalFlowExtractor(raft_model_path, device=device)
        self.detector = SimpleFlowDetector(self.flow_extractor, device=device)
        
        self.scanner = ActiveScanner(scan_distance=0.1, num_waypoints=5)
        
        # PnP estimator
        img_h = renderer.image_height
        img_w = renderer.image_width
        fov = renderer.fov
        
        self.camera_matrix = get_camera_matrix_from_fov(img_w, img_h, fov)
        self.pnp_estimator = WindowPnPEstimator(
            camera_matrix=self.camera_matrix,
            window_width=1.2,
            window_height=1.2
        )
        
        self.detected_windows = []
        self.window_count = 0
        self.scan_count = 0
        self.video_frames = []
        
        print("✓ Window Navigator initialized with PnP")
        print(f"  Image: {img_w}x{img_h}, FOV: {np.degrees(fov):.1f}°")
    
    def scan_for_window(self, current_pose, pose_history, scan_type='initial'):
        """Scan for window and estimate pose with PnP"""
        scan_label = f"{scan_type}_window{self.window_count}_scan{self.scan_count}"
        print(f"\n=== SCANNING FOR WINDOW ({scan_label}) ===")
        
        scan_waypoints = self.scanner.generate_scan_trajectory(current_pose)
        print(f"  Generated {len(scan_waypoints)} scan waypoints")
        
        scan_frames = []
        scan_poses = []
        
        for i, waypoint in enumerate(scan_waypoints):
            scan_pose = {
                'position': waypoint['position'].copy(),
                'rpy': waypoint['rpy'].copy()
            }
            
            rgb, _, _ = self.renderer.render(scan_pose['position'], scan_pose['rpy'])
            
            scan_frames.append(rgb)
            scan_poses.append(scan_pose)
            
            print(f"    ✓ Frame {i+1}/{len(scan_waypoints)}")
        
        # Detect window
        print("\n  Detecting window...")
        mask, center_2d, confidence, debug_info = self.detector.detect_window(scan_frames)
        
        os.makedirs('./log', exist_ok=True)
        viz_filename = f'./log/detection_{scan_label}.png'
        self.detector.visualize(debug_info, viz_filename)
        
        self.scan_count += 1
        
        if center_2d is None or confidence < 0.015:
            print(f"  ✗ No window (confidence: {confidence:.3f})")
            return None, scan_frames, None
        
        print(f"  ✓ Window at pixel {center_2d} (conf: {confidence:.3f})")
        
        # PnP POSE ESTIMATION
        print("\n  Estimating pose with PnP...")
        
        # CRITICAL: Use first frame as reference since flow computation uses first→last
        # The mask is generated from optical flow between frame 0 and frame N-1
        # So the mask coordinates are in the coordinate system of frame 0
        ref_idx = 0  # First frame
        ref_pose = scan_poses[ref_idx]
        ref_rgb = scan_frames[ref_idx]
        
        print(f"  Using frame {ref_idx} as reference (flow source frame)")
        
        # CRITICAL: mask is at ORIGINAL resolution (ref_rgb size)
        # Verify dimensions match
        print(f"  Mask shape: {mask.shape}")
        print(f"  Image shape: {ref_rgb.shape}")
        print(f"  Reference frame index: {ref_idx}/{len(scan_frames)-1}")
        
        if mask.shape[:2] != ref_rgb.shape[:2]:
            print(f"  ✗ DIMENSION MISMATCH! mask={mask.shape[:2]}, img={ref_rgb.shape[:2]}")
            return None, scan_frames, None
        
        # Additional sanity check: verify mask has nonzero pixels
        mask_pixels = np.sum(mask > 0.5)
        print(f"  Mask has {mask_pixels} nonzero pixels")
        if mask_pixels < 100:
            print(f"  ✗ Mask too small!")
            return None, scan_frames, None
        
        success, tvec_cam, rvec_cam, corners_2d = self.pnp_estimator.estimate_pose(mask)
        
        if not success:
            print(f"  ✗ PnP failed")
            return None, scan_frames, None
        
        # Visualize
        pnp_viz = f'./log/pnp_{scan_label}.png'
        self.pnp_estimator.visualize_pnp_result(ref_rgb, corners_2d, tvec_cam, rvec_cam, mask=mask, save_path=pnp_viz)
        
        print(f"  Camera frame: t={tvec_cam}, dist={np.linalg.norm(tvec_cam):.2f}m")
        
        # Transform to NED
        window_pos_ned, window_rpy_ned = self.pnp_estimator.transform_to_ned(
            tvec_cam, rvec_cam, ref_pose
        )
        
        print(f"  NED: pos={window_pos_ned}")
        print(f"       rpy(deg)={np.degrees(window_rpy_ned)}")
        
        # Collision check
        if doesItCollide(window_pos_ned):
            print(f"  ✗ Position collides, adjusting...")
            
            for adj_name, scale in [('closer', 0.8), ('further', 1.2), ('further2', 1.5)]:
                adjusted = ref_pose['position'] + (window_pos_ned - ref_pose['position']) * scale
                
                if not doesItCollide(adjusted):
                    print(f"  ✓ Adjusted ({adj_name}): {adjusted}")
                    window_pos_ned = adjusted
                    break
            else:
                print(f"  ✗ No collision-free position")
                return None, scan_frames, None
        
        return window_pos_ned, scan_frames, center_2d
    
    def align_with_window(self, current_pose, window_center_2d, pose_history):
        """Align drone with window center"""
        print("\n=== ALIGNING WITH WINDOW CENTER ===")
        
        rgb, _, _ = self.renderer.render(current_pose['position'], current_pose['rpy'])
        img_h, img_w = rgb.shape[:2]
        img_center_x = img_w / 2
        
        self.record_frame(rgb, pose=current_pose, annotation="Pre-alignment")
        
        window_x = window_center_2d[0]
        offset_x = window_x - img_center_x
        offset_ratio = offset_x / img_w
        
        print(f"  Window X: {window_x:.0f}, Center: {img_center_x:.0f}")
        print(f"  Offset: {offset_x:.0f}px ({offset_ratio:+.3f})")
        
        alignment_threshold = 0.05
        
        if abs(offset_ratio) < alignment_threshold:
            print(f"  ✓ Already centered")
            return current_pose, True
        
        current_yaw_deg = np.degrees(current_pose['rpy'][2])
        
        if offset_ratio > 0.1:
            yaw_adjustment_deg = 10
            if abs(current_yaw_deg + yaw_adjustment_deg) > 15:
                yaw_adjustment_deg = max(0, 15 - current_yaw_deg)
            print(f"  → Right by {yaw_adjustment_deg:.1f}°")
        elif offset_ratio < -0.1:
            yaw_adjustment_deg = -10
            if abs(current_yaw_deg + yaw_adjustment_deg) > 15:
                yaw_adjustment_deg = min(0, -15 - current_yaw_deg)
            print(f"  → Left by {abs(yaw_adjustment_deg):.1f}°")
        else:
            yaw_adjustment_deg = 0
            print(f"  → Lateral only")
        
        new_pose = current_pose.copy()
        
        if yaw_adjustment_deg != 0:
            target_yaw = wrap_angle(current_pose['rpy'][2] + np.radians(yaw_adjustment_deg))
            
            result = goToWaypoint_yaw(
                current_pose, target_yaw,
                pose_history=pose_history,
                action=f'YAW_ALIGN_W{self.window_count}',
                navigator=self
            )
            
            if result == -1:
                return -1, False
            
            new_pose = result
            print(f"  ✓ Yaw complete")
        
        lateral_distance = abs(offset_ratio) * 0.15
        
        if lateral_distance > 0.02:
            local_y_movement = lateral_distance if offset_ratio > 0 else -lateral_distance
            
            current_yaw = new_pose['rpy'][2]
            drone_right_ned = np.array([
                np.sin(current_yaw),
                np.cos(current_yaw),
                0.0
            ])
            
            move_vector_ned = drone_right_ned * local_y_movement
            target_pos_ned = new_pose['position'] + move_vector_ned
            
            print(f"  Lateral: {local_y_movement:+.3f}m")
            
            if not doesItCollide(target_pos_ned):
                result = goToWaypoint(
                    new_pose, target_pos_ned,
                    velocity=0.05,
                    pose_history=pose_history,
                    action=f'LATERAL_ALIGN_W{self.window_count}',
                    maintain_orientation=False,
                    navigator=self
                )
                
                if result != -1:
                    new_pose = result
                    print(f"  ✓ Lateral complete")
            else:
                print(f"  ✗ Target collides")
        
        return new_pose, False
    
    def navigate_through_window(self, current_pose, window_3d_pos, pose_history):
        """Navigate through window"""
        print(f"\n=== NAVIGATING THROUGH WINDOW {self.window_count + 1} ===")
        
        direction = window_3d_pos - current_pose['position']
        distance = np.linalg.norm(direction)
        
        print(f"  Current: {current_pose['position']}")
        print(f"  Window: {window_3d_pos}")
        print(f"  Distance: {distance:.3f}m")
        
        approach_distance = 0.8
        
        if distance > approach_distance:
            direction_norm = direction / distance
            approach_point = window_3d_pos - direction_norm * approach_distance
            
            if doesItCollide(approach_point):
                print(f"  ✗ Approach collides")
                return -1
            
            result = goToWaypoint(
                current_pose, approach_point,
                velocity=0.08,
                pose_history=pose_history,
                action=f'APPROACH_W{self.window_count}',
                lock_roll_pitch=True,
                navigator=self
            )
            
            if result == -1:
                return -1
            
            current_pose = result
            print(f"  ✓ Approached")
        
        # Verification
        print(f"\n  Verification scan...")
        verified_window, _, verified_center = self.scan_for_window(
            current_pose, pose_history, scan_type='verification'
        )
        
        if verified_window is None:
            print(f"  ✗ Verification failed")
            return -1
        
        print(f"  ✓ Verified")
        
        # Pass through
        through_distance = 1.5
        direction_norm = direction / distance
        through_point = window_3d_pos + direction_norm * through_distance
        
        if doesItCollide(through_point):
            print(f"  ✗ Through point collides")
            return -1
        
        result = goToWaypoint(
            current_pose, through_point,
            velocity=0.1,
            pose_history=pose_history,
            action=f'PASS_THROUGH_W{self.window_count}',
            lock_roll_pitch=True,
            navigator=self
        )
        
        if result == -1:
            return -1
        
        current_pose = result
        print(f"  ✓ Passed through")
        
        # Reset yaw
        result = goToWaypoint_yaw(
            current_pose, 0.0,
            pose_history=pose_history,
            action=f'YAW_RESET_W{self.window_count}',
            navigator=self
        )
        
        if result != -1:
            current_pose = result
        
        self.window_count += 1
        return current_pose
    
    def record_frame(self, rgb_frame, mask=None, frame_id=None, pose=None, annotation=None):
        """Save frame with pose overlay"""
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
            frame = rgb_frame.copy()
        
        if pose is not None:
            pos = pose['position']
            rpy = pose['rpy']
            rpy_deg = np.degrees(rpy)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            color = (0, 255, 255)
            bg_color = (0, 0, 0)
            
            text_lines = [
                f"XYZ: [{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}]",
                f"RPY: [{rpy_deg[0]:+.1f}, {rpy_deg[1]:+.1f}, {rpy_deg[2]:+.1f}]",
                f"Win: {self.window_count}"
            ]
            
            if annotation:
                text_lines.append(f"{annotation}")
            
            y_offset = 25
            for i, text in enumerate(text_lines):
                y_pos = y_offset + i * 30
                
                (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                
                cv2.rectangle(frame,
                            (5, y_pos - text_h - 3),
                            (10 + text_w, y_pos + baseline + 3),
                            bg_color, -1)
                
                cv2.putText(frame, text, (8, y_pos), font, font_scale,
                          color, thickness, cv2.LINE_AA)
        
        self.video_frames.append(frame)
        
        if frame_id is None:
            frame_id = len(self.video_frames) - 1
        
        frame_path = f'./log/frames/frame_{frame_id:04d}.png'
        os.makedirs('./log/frames', exist_ok=True)
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    def save_frames_summary(self):
        num_frames = len(self.video_frames)
        if num_frames > 0:
            print(f"\n✓ Saved {num_frames} frames to ./log/frames/")


def goToWaypoint_yaw(currentPose, target_yaw, pose_history=None, action='YAW', navigator=None):
    """Rotate in place to target yaw"""
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
            print(f'  ⚠ Position drift: {pos_error:.3f}m')

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
                                 annotation=f"{action} (yaw={np.degrees(temp_yaw):.1f}°)")
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
    """Navigate to waypoint"""
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

    if doesItCollide(target_position):
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
    
    check_stride = max(1, len(trajectory_points) // 50)
    for i in range(0, len(trajectory_points), check_stride):
        if doesItCollide(trajectory_points[i]):
            return -1

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

    for i, t in enumerate(time_points):
        control_input = controller.compute_control(state, t)
        current_pos = state[0:3]

        if doesItCollide(current_pos):
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


def main(renderer):
    os.makedirs('./log', exist_ok=True)
    os.makedirs('./log/frames', exist_ok=True)
    
    import glob
    for f in glob.glob('./log/*.png'):
        try:
            os.remove(f)
        except:
            pass
    
    pose_history = []
    
    print("\n" + "="*60)
    print("DRONE RACING - WITH PnP POSE ESTIMATION")
    print("="*60 + "\n")

    navigator = WindowNavigator(renderer, device='cuda')
    
    currentPose = {
        'position': np.array([0.0, 0.0, 0.0]),
        'rpy': np.radians([0.0, 0.0, 0.0])
    }
    
    pose_history.append({
        'step': 0,
        'action': 'INITIAL',
        'position': currentPose['position'].copy(),
        'rpy_deg': np.degrees(currentPose['rpy']),
        'rpy_rad': currentPose['rpy'].copy()
    })
    
    if doesItCollide(currentPose['position']):
        return -1
    
    rgb, _, _ = renderer.render(currentPose['position'], currentPose['rpy'])
    navigator.record_frame(rgb, pose=currentPose, annotation="START")
    
    print("\n" + "="*60)
    print("PHASE 1: FORWARD NAVIGATION")
    print("="*60)
    
    max_windows = 3
    
    for window_num in range(max_windows):
        print(f"\n{'='*60}")
        print(f"WINDOW {window_num + 1}")
        print(f"{'='*60}")
        
        window_3d_pos, scan_frames, window_center_2d = navigator.scan_for_window(
            currentPose, pose_history, scan_type='initial'
        )
        
        if window_3d_pos is None:
            print(f"  ✗ Detection/PnP failed")
            break
        
        print(f"  ✓ Window at: {window_3d_pos}")
        
        max_alignment_attempts = 3
        aligned = False
        
        for attempt in range(max_alignment_attempts):
            currentPose, aligned = navigator.align_with_window(
                currentPose, window_center_2d, pose_history
            )
            
            if currentPose == -1:
                break
            
            if aligned:
                break
            
            _, _, window_center_2d = navigator.scan_for_window(
                currentPose, pose_history, scan_type=f'align_check_{attempt}'
            )
            
            if window_center_2d is None:
                break
        
        result = navigator.navigate_through_window(
            currentPose, window_3d_pos, pose_history
        )
        
        if result == -1:
            break
        
        currentPose = result
    
    with open('./log/pose_history.json', 'w') as f:
        json.dump(pose_history, f, indent=2, default=str)
    
    navigator.save_frames_summary()
    
    print("\n" + "="*60)
    print("COMPLETE")
    print(f"  Windows: {navigator.window_count}")
    print("="*60 + "\n")


if __name__ == "__main__":
    config_path = "../data/P5_colmap_splat/P5_colmap/splatfacto/2025-11-17_130359/config.yml"
    json_path = "../data/render_settings/render_settings.json"

    renderer = SplatRenderer(config_path, json_path)
    main(renderer)