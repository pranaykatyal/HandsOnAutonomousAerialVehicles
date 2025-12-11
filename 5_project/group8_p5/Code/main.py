"""
Drone Racing Navigation System - WITH PnP INTEGRATION
[DONE] PnP-based window pose estimation
[DONE] Yaw changes use goToWaypoint (not instant)
[DONE] Coordinate transformation fixed
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
import torch

from window_detector import OpticalFlowExtractor, SimpleFlowDetector, ActiveScanner
from window_pnp import WindowPnPEstimator, get_camera_matrix_from_fov
import os
import json


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class WindowNavigator:
    """Manages window detection, alignment, and navigation with PnP pose estimation"""
    
    # CRITICAL: Map bounds for Gaussian splat environment
    MAP_Y_LIMIT = 2.0   # East/West
    MAP_Z_LIMIT = 0.5   # Down/Up
    MAP_X_LIMIT = 10.0  # North/South (generous)
    
    def __init__(self, renderer, device='cuda'):
        self.renderer = renderer
        self.device = device
        
        # Initialize optical flow detector
        raft_model_path = './RAFT/models/raft-things.pth'
        self.flow_extractor = OpticalFlowExtractor(raft_model_path, device=device)
        self.detector = SimpleFlowDetector(self.flow_extractor, device=device)
        
        # SCALED DOWN but not too small: 0.08m gives ±0.02m steps
        # This provides enough parallax while staying within ±2m Y bounds
        self.scanner = ActiveScanner(scan_distance=0.08, num_waypoints=5)
        
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
        
        print("âœ“ Window Navigator initialized with PnP")
        print(f"  Image: {img_w}x{img_h}, FOV: {np.degrees(fov):.1f}Â°")
        print(f"  Map bounds: X=+/-{self.MAP_X_LIMIT}m, Y=+/-{self.MAP_Y_LIMIT}m, Z=+/-{self.MAP_Z_LIMIT}m")
    
    def is_position_in_bounds(self, pos):
        """Check if position is within map bounds"""
        return (abs(pos[0]) <= self.MAP_X_LIMIT and
                abs(pos[1]) <= self.MAP_Y_LIMIT and
                abs(pos[2]) <= self.MAP_Z_LIMIT)
    
    def clip_position_to_bounds(self, pos):
        """Clip position to map bounds"""
        return np.array([
            np.clip(pos[0], -self.MAP_X_LIMIT, self.MAP_X_LIMIT),
            np.clip(pos[1], -self.MAP_Y_LIMIT, self.MAP_Y_LIMIT),
            np.clip(pos[2], -self.MAP_Z_LIMIT, self.MAP_Z_LIMIT)
        ])
    
    
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
            
            print(f"    âœ“ Frame {i+1}/{len(scan_waypoints)}")
        
        # Detect window
        print("\n  Detecting window...")
        mask, center_2d, confidence, debug_info = self.detector.detect_window(scan_frames)
        
        os.makedirs('./log', exist_ok=True)
        viz_filename = f'./log/detection_{scan_label}.png'
        self.detector.visualize(debug_info, viz_filename)
        
        self.scan_count += 1
        
        if center_2d is None or confidence < 0.005:  # LOWERED: Was 0.015, reduced for smaller scan distances
            print(f"  [ERROR] No window detected")
            print(f"    Confidence: {confidence:.3f} (threshold: 0.005)")
            print(f"    Mask pixels: {np.sum(mask > 0.5) if mask is not None else 0:.0f}")
            print(f"    Center: {center_2d}")
            return None, scan_frames, None
        
        print(f"  âœ“ Window at pixel {center_2d} (conf: {confidence:.3f})")
        
        # PnP POSE ESTIMATION
        print("\n  Estimating pose with PnP...")
        
        # CRITICAL: Use first frame as reference since flow computation uses firstâ†’last
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
            print(f"  [ERROR] DIMENSION MISMATCH! mask={mask.shape[:2]}, img={ref_rgb.shape[:2]}")
            return None, scan_frames, None
        
        # Additional sanity check: verify mask has nonzero pixels
        mask_pixels = np.sum(mask > 0.5)
        print(f"  Mask has {mask_pixels} nonzero pixels")
        if mask_pixels < 100:
            print(f"  [ERROR] Mask too small!")
            return None, scan_frames, None
        
        success, tvec_cam, rvec_cam, corners_2d = self.pnp_estimator.estimate_pose(mask)
        
        if not success:
            print(f"  [ERROR] PnP failed")
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
            print(f"  [ERROR] Position collides, adjusting...")
            
            for adj_name, scale in [('closer', 0.8), ('further', 1.2), ('further2', 1.5)]:
                adjusted = ref_pose['position'] + (window_pos_ned - ref_pose['position']) * scale
                
                if not doesItCollide(adjusted):
                    print(f"  âœ“ Adjusted ({adj_name}): {adjusted}")
                    window_pos_ned = adjusted
                    break
            else:
                print(f"  [ERROR] No collision-free position")
                return None, scan_frames, None
        
        # CRITICAL: Check if window position is within map bounds
        if not self.is_position_in_bounds(window_pos_ned):
            print(f"  [ERROR] Window position outside map bounds!")
            print(f"    Position: X={window_pos_ned[0]:.2f}, Y={window_pos_ned[1]:.2f}, Z={window_pos_ned[2]:.2f}")
            print(f"    Limits: X=+/-{self.MAP_X_LIMIT}, Y=+/-{self.MAP_Y_LIMIT}, Z=+/-{self.MAP_Z_LIMIT}")
            return None, scan_frames, None
        
        # Ensure frames are saved in log/frames with proper numbering
        os.makedirs('./log/frames', exist_ok=True)
        for idx, frame in enumerate(scan_frames):
            frame_path = f'./log/frames/frame_{idx:03d}.png'
            cv2.imwrite(frame_path, frame[:, :, ::-1])  # Convert RGB to BGR for saving
            print(f"  [OK] Saved frame {idx+1}/{len(scan_frames)} to {frame_path}")
        
        return window_pos_ned, scan_frames, center_2d
    
    def _debug_visualize_alignment_flow(self, current_pose, label="debug"):
        """
        DEBUG: Compute and visualize optical flow from current position
        Creates side-by-side: Current Frame | Flow Magnitude | Detection Mask
        """
        print(f"\n  DEBUG: Computing flow at {label}")
        
        # Generate mini scan (3 frames for quick flow computation)
        scan_waypoints = self.scanner.generate_scan_trajectory(current_pose)
        
        debug_frames = []
        for i in range(min(3, len(scan_waypoints))):
            waypoint = scan_waypoints[i]
            rgb, _, _ = self.renderer.render(waypoint['position'], waypoint['rpy'])
            debug_frames.append(rgb)
        
        if len(debug_frames) < 2:
            print("  DEBUG: Not enough frames")
            return None, None
        
        # Downsample for flow computation
        frame1_small = cv2.resize(debug_frames[0], (512, 512))
        frame2_small = cv2.resize(debug_frames[-1], (512, 512))
        
        # Compute optical flow
        frame1_torch = torch.from_numpy(frame1_small).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        frame2_torch = torch.from_numpy(frame2_small).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        frame1_torch = frame1_torch.to(self.device)
        frame2_torch = frame2_torch.to(self.device)
        
        with torch.no_grad():
            flow = self.flow_extractor.compute_flow(frame1_torch, frame2_torch)
            u = flow[0, 0]
            v = flow[0, 1]
            flow_mag = torch.sqrt(u**2 + v**2)
            Xi_np = flow_mag.cpu().numpy()
        
        print(f"  DEBUG: Flow range [{Xi_np.min():.1f}, {Xi_np.max():.1f}], mean {Xi_np.mean():.1f}")
        
        # Run detection on these frames to get mask
        mask, _, _, _ = self.detector.detect_window(debug_frames)
        
        # Return flow and mask for visualization
        return Xi_np, mask
    
    
    def visual_servo_to_window(self, current_pose, window_3d_pos, window_rpy_ned, pose_history):
        """
        Visual servoing approach using PnP pose tracking
        
        1. Orient yaw to face window
        2. Servo Y/Z to center window (±5px accuracy)
        3. Move forward when centered
        
        Args:
            current_pose: Current drone pose
            window_3d_pos: Window 3D position (NED)
            window_rpy_ned: Window orientation (NED)
            pose_history: Pose history list
            
        Returns:
            final_pose: Pose when reached window or -1 if failed
        """
        print("\n=== VISUAL SERVOING TO WINDOW ===")
        
        # Calculate initial yaw error
        vec_to_window = window_3d_pos - current_pose['position']
        desired_yaw = np.arctan2(vec_to_window[1], vec_to_window[0])  # atan2(East, North)
        current_yaw = current_pose['rpy'][2]
        yaw_error = wrap_angle(desired_yaw - current_yaw)

        # STEP 1: Align yaw to face window
        print("\n  STEP 1: Aligning yaw to face window...")
        yaw_tolerance = np.radians(1.0)  # 1° tolerance
        yaw_step = np.radians(1.0)  # Increment yaw by 1° per iteration

        while abs(yaw_error) > yaw_tolerance:
            # Incrementally adjust yaw
            step_yaw = np.clip(yaw_error, -yaw_step, yaw_step)
            current_pose['rpy'][2] += step_yaw  # Adjust yaw only
            yaw_error = wrap_angle(desired_yaw - current_pose['rpy'][2])

            print(f"  Adjusting yaw: Current={np.degrees(current_pose['rpy'][2]):.1f}°, Target={np.degrees(desired_yaw):.1f}°, Error={np.degrees(yaw_error):.1f}°")

        print("  [OK] Yaw aligned to window")

        # STEP 2: Align X and Y to center the window
        print("\n  STEP 2: Aligning X and Y to center window...")
        position_tolerance = 0.05  # 5 cm tolerance

        while True:
            # Compute position error
            position_error = window_3d_pos[:2] - current_pose['position'][:2]  # X and Y only
            error_magnitude = np.linalg.norm(position_error)

            if error_magnitude < position_tolerance:
                print("  [OK] Position aligned to window")
                break

            # Move towards target position
            step_position = 0.1 * position_error / error_magnitude  # Scale step size
            current_pose['position'][:2] += step_position  # Adjust X and Y only

            print(f"  Adjusting position: Current={current_pose['position'][:2]}, Target={window_3d_pos[:2]}, Error={position_error}")
        
        # STEP 3: Move forward if not too close
        distance_threshold = 0.5  # Stop servoing when this close (meters)
        max_distance = 20.0  # Abort if window gets too far (something's wrong)
        distance_to_window = np.linalg.norm(window_3d_pos - current_pose['position'])
        
        if distance_to_window > distance_threshold:
            print(f"\n  STEP 3: Moving forward...")
            forward_distance = min(0.3, distance_to_window - 0.3)
            
            # Compute forward position in NED
            current_yaw = current_pose['rpy'][2]
            forward_vec_ned = np.array([
                np.cos(current_yaw),  # North
                np.sin(current_yaw),  # East
                0.0                   # Down (maintain altitude)
            ]) * forward_distance
            
            target_pos_ned = current_pose['position'] + forward_vec_ned
            
            if doesItCollide(target_pos_ned):
                print(f"  [ERROR] Forward position collides")
                return -1
            
            # Use safe_update_position to validate the target position
            updated_pose = self.safe_update_position(current_pose, target_pos_ned)
            if updated_pose is None:
                return -1

            # Proceed with the movement if no collision
            result = goToWaypoint(
                current_pose, target_pos_ned,
                velocity=0.08,
                pose_history=pose_history,
                action=f'FORWARD_W{self.window_count}',
                lock_roll_pitch=True,
                navigator=self
            )

            if result == -1:
                return -1

            current_pose = result
            print(f"  [OK] Moved forward {forward_distance:.2f}m")
        else:
            print(f"  [OK] Close enough to window ({distance_to_window:.2f}m)")
        
        return current_pose
    
    def navigate_through_window(self, current_pose, window_3d_pos, pose_history):
        """Navigate through window"""
        print(f"\n=== NAVIGATING THROUGH WINDOW {self.window_count + 1} ===")
        
        direction = window_3d_pos - current_pose['position']
        distance = np.linalg.norm(direction)
        
        print(f"  Current: {current_pose['position']}")
        print(f"  Window: {window_3d_pos}")
        print(f"  Distance: {distance:.3f}m")
        
        approach_distance = 0.3  # SCALED DOWN for +/-2m Y range
        
        if distance > approach_distance:
            direction_norm = direction / distance
            approach_point = window_3d_pos - direction_norm * approach_distance
            
            # BOUNDS CHECK
            if not self.is_position_in_bounds(approach_point):
                print(f"  [ERROR] Approach point outside map bounds: {approach_point}")
                approach_point = self.clip_position_to_bounds(approach_point)
                print(f"  [WARN] Clipped to: {approach_point}")
            
            if doesItCollide(approach_point):
                print(f"  [ERROR] Approach collides")
                return -1
            
            result = goToWaypoint(
                current_pose, approach_point,
                velocity=0.05,  # SCALED DOWN
                pose_history=pose_history,
                action=f'APPROACH_W{self.window_count}',
                lock_roll_pitch=True,
                navigator=self
            )
            
            if result == -1:
                return -1
            
            current_pose = result
            print(f"  âœ“ Approached")
        
        # Verification
        print(f"\n  Verification scan...")
        verified_window, _, verified_center = self.scan_for_window(
            current_pose, pose_history, scan_type='verification'
        )
        
        if verified_window is None:
            print(f"  [ERROR] Verification failed")
            return -1
        
        print(f"  âœ“ Verified")
        
        # Pass through
        through_distance = 0.5  # SCALED DOWN
        direction_norm = direction / distance
        through_point = window_3d_pos + direction_norm * through_distance
        
        # BOUNDS CHECK
        if not self.is_position_in_bounds(through_point):
            print(f"  [ERROR] Through point outside map bounds: {through_point}")
            through_point = self.clip_position_to_bounds(through_point)
            print(f"  [WARN] Clipped to: {through_point}")
        
        if doesItCollide(through_point):
            print(f"  [ERROR] Through point collides")
            return -1
        
        result = goToWaypoint(
            current_pose, through_point,
            velocity=0.05,  # SCALED DOWN
            pose_history=pose_history,
            action=f'PASS_THROUGH_W{self.window_count}',
            lock_roll_pitch=True,
            navigator=self
        )
        
        if result == -1:
            return -1
        
        current_pose = result
        print(f"  âœ“ Passed through")
        
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
    
    def record_frame(self, rgb_frame, mask=None, frame_id=None, pose=None, annotation=None, flow_mag=None, flow_mask=None):
        """
        Save frame with pose overlay and optional flow visualization
        
        If flow_mag or flow_mask provided, creates 3-panel side-by-side:
        [Current Frame | Flow Magnitude | Detection Mask]
        
        Args:
            rgb_frame: RGB image
            mask: Detection mask (optional)
            frame_id: Frame number (optional)
            pose: Drone pose dict (optional)
            annotation: Text annotation (optional)
            flow_mag: Optical flow magnitude array (optional)
            flow_mask: Detection mask from flow (optional)
        """
        if len(self.video_frames) == 0:
            import glob
            for f in glob.glob('./log/frames/*.png'):
                try:
                    os.remove(f)
                except:
                    pass
        
        if frame_id is None:
            frame_id = len(self.video_frames)
        
        # CREATE 3-PANEL VISUALIZATION if flow data provided
        if flow_mag is not None or flow_mask is not None:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 3, figsize=(24, 8))
            
            # Panel 1: Current frame with pose overlay
            display_frame = rgb_frame.copy()
            if pose is not None:
                pos = pose['position']
                rpy_deg = np.degrees(pose['rpy'])
                
                # Add text overlay
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                thickness = 3
                color = (0, 255, 255)
                bg_color = (0, 0, 0)
                
                text_lines = [
                    f"XYZ: [{pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f}]",
                    f"RPY: [{rpy_deg[0]:+.1f}, {rpy_deg[1]:+.1f}, {rpy_deg[2]:+.1f}]",
                    f"Win: {self.window_count}"
                ]
                if annotation:
                    text_lines.append(f"{annotation}")
                
                y_offset = 40
                for i, text in enumerate(text_lines):
                    y_pos = y_offset + i * 50
                    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                    cv2.rectangle(display_frame, (10, y_pos - text_h - 5),
                                (15 + text_w, y_pos + baseline + 5), bg_color, -1)
                    cv2.putText(display_frame, text, (12, y_pos), font, font_scale,
                              color, thickness, cv2.LINE_AA)
            
            axes[0].imshow(display_frame)
            axes[0].set_title('Current View', fontsize=16, fontweight='bold')
            axes[0].axis('off')
            
            # Panel 2: Flow magnitude
            if flow_mag is not None:
                # Resize flow to match frame size for display
                if flow_mag.shape[:2] != rgb_frame.shape[:2]:
                    flow_display = cv2.resize(flow_mag, (rgb_frame.shape[1], rgb_frame.shape[0]))
                else:
                    flow_display = flow_mag
                    
                im = axes[1].imshow(flow_display, cmap='jet', vmin=0, vmax=35)
                axes[1].set_title(f'Optical Flow\nRange: [{flow_mag.min():.1f}, {flow_mag.max():.1f}], Mean: {flow_mag.mean():.1f}', 
                                 fontsize=16, fontweight='bold')
                axes[1].axis('off')
                cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
                cbar.set_label('Flow Magnitude (pixels)', fontsize=12)
                
                # Add threshold markers
                axes[1].axhline(y=flow_display.shape[0]//2, color='lime', linestyle='--', 
                              linewidth=2, alpha=0.7, label='Target: 6-12')
            else:
                axes[1].text(0.5, 0.5, 'No Flow Data', ha='center', va='center', 
                           fontsize=20, transform=axes[1].transAxes)
                axes[1].set_title('Optical Flow', fontsize=16, fontweight='bold')
                axes[1].axis('off')
            
            # Panel 3: Detection mask
            if flow_mask is not None and flow_mask.max() > 0:
                # Resize mask to match frame
                if flow_mask.shape[:2] != rgb_frame.shape[:2]:
                    mask_display = cv2.resize(flow_mask, (rgb_frame.shape[1], rgb_frame.shape[0]))
                else:
                    mask_display = flow_mask
                    
                axes[2].imshow(mask_display, cmap='gray', vmin=0, vmax=1)
                mask_pixels = np.sum(mask_display > 0.5)
                mask_pct = (mask_pixels / mask_display.size) * 100
                axes[2].set_title(f'Detection Mask\nPixels: {mask_pixels} ({mask_pct:.1f}%)', 
                                fontsize=16, fontweight='bold')
            else:
                axes[2].text(0.5, 0.5, 'No Detection', ha='center', va='center', 
                           fontsize=20, color='red', transform=axes[2].transAxes)
                axes[2].set_title('Detection Mask', fontsize=16, fontweight='bold')
            axes[2].axis('off')
            
            plt.tight_layout()
            frame_path = f'./log/frames/frame_{frame_id:04d}.png'
            os.makedirs('./log/frames', exist_ok=True)
            plt.savefig(frame_path, dpi=120, bbox_inches='tight')
            plt.close()
            
            # Also save to video_frames for potential video creation
            # Read back the saved image
            saved_img = cv2.imread(frame_path)
            saved_img_rgb = cv2.cvtColor(saved_img, cv2.COLOR_BGR2RGB)
            self.video_frames.append(saved_img_rgb)
            
        else:
            # STANDARD SINGLE-PANEL (when no flow data)
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
                    cv2.rectangle(frame, (5, y_pos - text_h - 3),
                                (10 + text_w, y_pos + baseline + 3), bg_color, -1)
                    cv2.putText(frame, text, (8, y_pos), font, font_scale,
                              color, thickness, cv2.LINE_AA)
            
            self.video_frames.append(frame)
            frame_path = f'./log/frames/frame_{frame_id:04d}.png'
            os.makedirs('./log/frames', exist_ok=True)
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    def save_frames_summary(self):
        num_frames = len(self.video_frames)
        if num_frames > 0:
            print(f"\nâœ“ Saved {num_frames} frames to ./log/frames/")
    
    def safe_update_position(self, current_pose, target_pos_ned):
        """
        Updates the drone's position safely by checking for collisions.

        Args:
            current_pose: Current pose of the drone.
            target_pos_ned: Target position in NED coordinates.

        Returns:
            Updated pose if no collision, else None.
        """
        print("Checking collision for target position")
        if doesItCollide(target_pos_ned):
            print(f"  [ERROR] Target position collides: {target_pos_ned}")
            return None

        # Update the current pose with the new position
        updated_pose = current_pose.copy()
        updated_pose['position'] = target_pos_ned
        return updated_pose


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
            print(f'  [WARN]  Position drift: {pos_error:.3f}m')

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
            
            # Use the proper renderer for rendering frames
            rgb_temp, _, _ = navigator.renderer.render(current_pos, temp_rpy)

            # Overlay current pose on the top-right corner of the image BEFORE saving
            overlay_text = f"Pose: {current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}"
            cv2.putText(
                rgb_temp, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
            )

            # Save frames during yaw alignment
            os.makedirs('./log/frames', exist_ok=True)
            frame_path = f'./log/frames/frame_yaw_alignment_{i:03d}.png'
            cv2.imwrite(frame_path, rgb_temp[:, :, ::-1])  # Convert RGB to BGR for saving

            # Improved debug statement to confirm frame saving
            print(f"[DEBUG] Frame {i+1}/{len(time_points)} saved during yaw alignment: {frame_path}")

            # Update last capture time to ensure consistent saving
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

    # BOUNDS CHECK: Map limits Y=+/-2.0, Z=+/-1.0
    MAP_X_LIMIT, MAP_Y_LIMIT, MAP_Z_LIMIT = 10.0, 2.0, 1.0
    if (abs(target_position[0]) > MAP_X_LIMIT or
        abs(target_position[1]) > MAP_Y_LIMIT or
        abs(target_position[2]) > MAP_Z_LIMIT):
        print(f"  [ERROR] goToWaypoint target outside map bounds: {target_position}")
        return -1

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
            
            # If this is alignment motion, compute and save flow debug info
            flow_mag, flow_mask = None, None
            if 'ALIGN' in action or 'YAW' in action:
                flow_mag, flow_mask = navigator._debug_visualize_alignment_flow(
                    temp_pose, label=f"{action}_t{t:.1f}"
                )
            
            navigator.record_frame(rgb_temp, pose=temp_pose, annotation=f"{action}", 
                                  flow_mag=flow_mag, flow_mask=flow_mask)
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
            print(f"  [ERROR] Detection/PnP failed")
            break
        
        print(f"  âœ“ Window at: {window_3d_pos}")
        
        print(f"  [OK] Window detected at: {window_3d_pos}")
        
        # Get window orientation from last PnP estimate
        window_rpy_ned = np.array([0.0, 0.0, 0.0])  # Placeholder - assume perpendicular
        
        # STEP 2: Visual servoing (orient, center, approach)
        result = navigator.visual_servo_to_window(
            currentPose, window_3d_pos, window_rpy_ned, pose_history
        )
        
        if result == -1:
            print(f"  [ERROR] Visual servoing failed")
            break
        
        currentPose = result
        print(f"  [OK] Reached window via visual servoing")
        
        # STEP 3: Pass through window
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