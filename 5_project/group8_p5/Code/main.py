"""
Drone Racing Navigation System - WITH PnP INTEGRATION
[UPDATED] Projection-only visual servoing to avoid detector jumping to wrong windows
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
from navigation import goToWaypoint, goToWaypoint_yaw, wrap_angle
import os
import json


class WindowNavigator:
    """Manages window detection, alignment, and navigation with PnP pose estimation"""
    
    # CRITICAL: Map bounds for Gaussian splat environment
    MAP_Y_LIMIT = 2.0   # East/West
    MAP_Z_LIMIT = 0.5   # Down/Up
    MAP_X_LIMIT = 10.0  # North/South (generous)
    
    def __init__(self, renderer, device='cuda'):
        self.renderer = renderer
        self.device = device
        
        # Pose history for logging
        self.pose_history = []
        
        # PnP scale factor
        self.PNP_SCALE_FACTOR = 2.0
        
        # Initialize optical flow detector
        raft_model_path = './RAFT/models/raft-things.pth'
        self.flow_extractor = OpticalFlowExtractor(raft_model_path, device=device)
        self.detector = SimpleFlowDetector(self.flow_extractor, device=device)
        
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
        
        print("Window Navigator initialized with PnP")
        print(f"  Image: {img_w}x{img_h}, FOV: {np.degrees(fov):.1f} deg")
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
    
    def scan_for_window(self, current_pose, pose_history, scan_type='initial', target_pixel=None, yaw_hint=None):
        """Scan for window and estimate pose with PnP
        
        Args:
            yaw_hint: Initial yaw error (radians) to guide VERIFY selection
                     +ve = window on RIGHT, -ve = window on LEFT
        """
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
            
            print(f"    Frame {i+1}/{len(scan_waypoints)}")
        
        # Detect window
        print("\n  Detecting window...")
        
        # VERIFY mode: prefer LARGER component (closer window)
        # SCAN mode: use heuristic scoring (better quality)
        prefer_larger = scan_type.startswith('verify')
        mask, center_2d, confidence, debug_info = self.detector.detect_window(
            scan_frames, 
            prefer_larger=prefer_larger,
            yaw_hint=yaw_hint
        )

        # Extra debug: label all contours
        try:
            if mask is not None:
                mask_uint8_dbg = (mask * 255).astype(np.uint8)
                contours_dbg, _ = cv2.findContours(mask_uint8_dbg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours_dbg) > 0:
                    ref_img = scan_frames[0].copy()
                    overlay = ref_img.copy()
                    stats = []
                    for idx_c, c in enumerate(contours_dbg):
                        area = cv2.contourArea(c)
                        M = cv2.moments(c)
                        if M['m00'] == 0:
                            cx, cy = -1, -1
                        else:
                            cx = int(M['m10'] / M['m00'])
                            cy = int(M['m01'] / M['m00'])
                        stats.append((idx_c, area, (cx, cy)))
                        cv2.drawContours(overlay, [c.astype(np.int32)], -1, (200, 200, 255), -1)
                        cv2.putText(ref_img, f"{idx_c}", (cx+5, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
                        cv2.putText(ref_img, f"{idx_c}", (cx+5, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                        cv2.putText(ref_img, f"A:{int(area)}", (cx+5, cy+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

                    vis_labeled = cv2.addWeighted(ref_img, 0.7, overlay, 0.3, 0)
                    labeled_path = f"./log/detection_{scan_label}_labeled.png"
                    os.makedirs('./log', exist_ok=True)
                    cv2.imwrite(labeled_path, cv2.cvtColor(vis_labeled, cv2.COLOR_RGB2BGR))
                    print(f"  [DEBUG] Saved labeled contour visualization: {labeled_path}")
                    
                    # Create mask visualization
                    try:
                        mask_viz = cv2.cvtColor(mask_uint8_dbg, cv2.COLOR_GRAY2BGR)
                        for idx_c, c in enumerate(contours_dbg):
                            area = cv2.contourArea(c)
                            M = cv2.moments(c)
                            if M['m00'] == 0:
                                cx, cy = -1, -1
                            else:
                                cx = int(M['m10'] / M['m00'])
                                cy = int(M['m01'] / M['m00'])
                            cv2.drawContours(mask_viz, [c.astype(np.int32)], -1, (255, 200, 200), 2)
                            if cx >= 0 and cy >= 0:
                                cv2.circle(mask_viz, (cx, cy), 6, (0, 0, 255), -1)
                                cv2.putText(mask_viz, f"{idx_c}", (cx+8, cy+6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                                cv2.putText(mask_viz, f"A:{int(area)}", (cx+8, cy+28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                        mask_labeled_path = f"./log/detection_{scan_label}_mask_labeled.png"
                        cv2.imwrite(mask_labeled_path, cv2.cvtColor(mask_viz, cv2.COLOR_RGB2BGR))
                        print(f"  [DEBUG] Saved mask-labeled visualization: {mask_labeled_path}")
                    except Exception as e:
                        print(f"  [WARN] Failed to save mask-labeled visualization: {e}")
                    
                    print("  [DEBUG] Contour stats (idx, area, centroid):")
                    for s in stats:
                        print(f"    - {s[0]}: area={s[1]:.0f}, centroid={s[2]}")
        except Exception as e:
            print(f"  [WARN] Failed to create labeled contour visualization: {e}")

        # If target_pixel provided, prefer contour closest to it
        if target_pixel is not None and mask is not None:
            try:
                mask_uint8 = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) > 0:
                    tgt = np.array([float(target_pixel[0]), float(target_pixel[1])])
                    best_idx = None
                    best_dist = float('inf')
                    centroids = []
                    for i, c in enumerate(contours):
                        M = cv2.moments(c)
                        if M['m00'] == 0:
                            continue
                        cx = M['m10'] / M['m00']
                        cy = M['m01'] / M['m00']
                        centroids.append((cx, cy))
                        d = np.linalg.norm(tgt - np.array([cx, cy]))
                        if d < best_dist:
                            best_dist = d
                            best_idx = i

                    if best_idx is not None and best_dist < 500:
                        chosen = contours[best_idx]
                        chosen_centroid = centroids[best_idx]
                        print(f"  [INFO] Selecting contour closest to projected pixel {target_pixel}: dist={best_dist:.1f}px")
                        new_mask = np.zeros_like(mask_uint8)
                        cv2.drawContours(new_mask, [chosen], -1, 255, -1)
                        mask = (new_mask.astype(np.float32) / 255.0)
                        center_2d = (chosen_centroid[0], chosen_centroid[1])
                        if debug_info is not None:
                            debug_info['preferred_contour_selected'] = True
                    else:
                        print(f"  [INFO] No contour close enough to projected pixel (best_dist={best_dist:.1f})")
            except Exception as e:
                print(f"  [WARN] Targeted contour selection failed: {e}")
        
        os.makedirs('./log', exist_ok=True)
        viz_filename = f'./log/detection_{scan_label}.png'
        self.detector.visualize(debug_info, viz_filename)
        
        if os.path.exists(viz_filename):
            print(f"   SAVED DETECTION VISUALIZATION: {viz_filename} ")
            print(f"      File size: {os.path.getsize(viz_filename)} bytes")
        else:
            print(f"   FAILED TO SAVE: {viz_filename} ")
        
        self.scan_count += 1
        
        if center_2d is None or confidence < 0.005:
            print(f"  [ERROR] No window detected")
            print(f"    Confidence: {confidence:.3f} (threshold: 0.005)")
            print(f"    Mask pixels: {np.sum(mask > 0.5) if mask is not None else 0:.0f}")
            print(f"    Center: {center_2d}")
            return None, scan_frames, None, None
        
        print(f"  Window at pixel {center_2d} (conf: {confidence:.3f})")
        
        # PnP POSE ESTIMATION
        print("\n  Estimating pose with PnP...")
        
        ref_idx = 0
        ref_pose = scan_poses[ref_idx]
        ref_rgb = scan_frames[ref_idx]
        
        print(f"  Using frame {ref_idx} as reference")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Image shape: {ref_rgb.shape}")
        
        if mask.shape[:2] != ref_rgb.shape[:2]:
            print(f"  [ERROR] DIMENSION MISMATCH! mask={mask.shape[:2]}, img={ref_rgb.shape[:2]}")
            return None, scan_frames, None, None
        
        mask_pixels = np.sum(mask > 0.5)
        print(f"  Mask has {mask_pixels} nonzero pixels")
        if mask_pixels < 100:
            print(f"  [ERROR] Mask too small!")
            return None, scan_frames, None, None
        
        success, tvec_cam, rvec_cam, corners_2d = self.pnp_estimator.estimate_pose(mask)
        
        if not success:
            print(f"  [ERROR] PnP failed")
            return None, scan_frames, None, None
        
        # Visualize
        pnp_viz = f'./log/pnp_{scan_label}.png'
        self.pnp_estimator.visualize_pnp_result(ref_rgb, corners_2d, tvec_cam, rvec_cam, mask=mask, save_path=pnp_viz)
        
        if os.path.exists(pnp_viz):
            print(f"   SAVED PNP VISUALIZATION: {pnp_viz} ")
            print(f"      File size: {os.path.getsize(pnp_viz)} bytes")
        else:
            print(f"   FAILED TO SAVE: {pnp_viz} ")
        
        print(f"  Camera frame: t={tvec_cam}, dist={np.linalg.norm(tvec_cam):.2f}m")
        
        # Transform to NED with SCALE FACTOR
        window_pos_ned, window_rpy_ned = self.pnp_estimator.transform_to_ned(
            tvec_cam * self.PNP_SCALE_FACTOR,
            rvec_cam, 
            ref_pose
        )
        
        print(f"  NED (scaled by {self.PNP_SCALE_FACTOR}): pos={window_pos_ned}")
        print(f"       rpy(deg)={np.degrees(window_rpy_ned)}")
        
        # Collision check
        if doesItCollide(window_pos_ned):
            print(f"  [ERROR] Position collides, adjusting...")
            
            for adj_name, scale in [('closer', 0.8), ('further', 1.2), ('further2', 1.5)]:
                adjusted = ref_pose['position'] + (window_pos_ned - ref_pose['position']) * scale
                
                if not doesItCollide(adjusted):
                    print(f"  Adjusted ({adj_name}): {adjusted}")
                    window_pos_ned = adjusted
                    break
            else:
                print(f"  [ERROR] No collision-free position")
                return None, scan_frames, None, None
        
        # Check if window position is within map bounds
        if not self.is_position_in_bounds(window_pos_ned):
            print(f"  [ERROR] Window position outside map bounds!")
            print(f"    Position: X={window_pos_ned[0]:.2f}, Y={window_pos_ned[1]:.2f}, Z={window_pos_ned[2]:.2f}")
            print(f"    Limits: X=+/-{self.MAP_X_LIMIT}, Y=+/-{self.MAP_Y_LIMIT}, Z=+/-{self.MAP_Z_LIMIT}")
            return None, scan_frames, None, None
        
        # Save frames
        for idx, (frame, pose) in enumerate(zip(scan_frames, scan_poses)):
            self.record_frame(frame, pose=pose, annotation=f"SCAN {scan_type}")
            print(f"  [OK] Saved scan frame {idx+1}/{len(scan_frames)}")
        
        return window_pos_ned, scan_frames, center_2d, corners_2d
    
    def estimate_depth_from_height(self, corners_2d, real_height=0.067):
        """
        Estimate depth Z using known real-world height of window
        
        Args:
            corners_2d: (4, 2) array [BL, BR, TR, TL]
            real_height: Real window height in splat units
            
        Returns:
            Z_cam: Depth in splat units, or None
        """
        if corners_2d is None or len(corners_2d) != 4:
            return None
        
        try:
            # Measure pixel height
            top_y = 0.5 * (corners_2d[2, 1] + corners_2d[3, 1])
            bot_y = 0.5 * (corners_2d[0, 1] + corners_2d[1, 1])
            h_px = abs(bot_y - top_y)
            
            if h_px < 5:
                print(f"    [WARN] Pixel height too small: {h_px:.1f}px")
                return None
            
            # Similar triangles: Z = f * H_real / h_px
            fx = self.pnp_estimator.camera_matrix_original[0, 0]
            Z_cam = (fx * real_height) / h_px
            
            print(f"    Depth from height: h_px={h_px:.1f}px -> Z={Z_cam:.3f} splat units")
            return Z_cam
        except Exception as e:
            print(f"    [WARN] Depth estimation failed: {e}")
            return None
    
    def visual_servo_to_window(self, current_pose, window_3d_pos, window_rpy_ned, pose_history, initial_corners_2d=None, scan_frames=None):
        """
        PROJECTION-BASED Visual Servoing:
        1. Yaw alignment using initial window pose
        2. Lateral/vertical centering using ONLY projection (no re-detection!)
        3. Move forward when centered
        
        Args:
            current_pose: Current drone pose
            window_3d_pos: Window 3D position (NED) from INITIAL detection
            window_rpy_ned: Window orientation (NED)
            pose_history: Pose history list
            initial_corners_2d: Initial PnP corners (4, 2) array
            scan_frames: Scan frames from detection phase
            
        Returns:
            final_pose: Pose when reached window or -1 if failed
        """
        print("\n" + "="*70)
        print("PROJECTION-BASED VISUAL SERVOING (No Re-detection)")
        print("="*70)
        
        # Calculate initial yaw error
        vec_to_window = window_3d_pos - current_pose['position']
        desired_yaw = np.arctan2(vec_to_window[1], vec_to_window[0])
        current_yaw = current_pose['rpy'][2]
        yaw_error = wrap_angle(desired_yaw - current_yaw)

        # ================================================================
        # PHASE 1: YAW + HEIGHT ALIGNMENT (Decoupled corrections)
        # ================================================================
        print("\n[PHASE 1] Yaw + Height Alignment")
        print("  Strategy: Correct yaw and Z simultaneously (they're independent!)")
        yaw_tolerance = np.radians(1.0)
        yaw_step = np.radians(1.0)
        z_tolerance = 0.01  # 1cm tolerance in Z
        
        yaw_iter = 0
        max_yaw_iterations = 30
        
        # Get initial depth estimate for Z correction
        Z_cam_initial = self.estimate_depth_from_height(initial_corners_2d, real_height=0.067)
        
        while abs(yaw_error) > yaw_tolerance and yaw_iter < max_yaw_iterations:
            # YAW correction
            step_yaw = np.clip(yaw_error, -yaw_step, yaw_step)
            current_pose['rpy'][2] += step_yaw
            
            # Z (HEIGHT) correction - simultaneous with yaw!
            proj_pixel = self.pnp_estimator.project_window_to_pixel(window_3d_pos, current_pose)
            
            if proj_pixel is not None:
                rgb, _, _ = self.renderer.render(current_pose['position'], current_pose['rpy'])
                img_h, img_w = rgb.shape[:2]
                img_center_y = img_h / 2
                
                # Vertical error only (for Z correction)
                error_y_px = proj_pixel[1] - img_center_y
                
                # Get depth
                cam_vec = self.pnp_estimator.get_camera_vector(window_3d_pos, current_pose)
                if cam_vec is not None and cam_vec[2] > 0:
                    Z_cam = cam_vec[2]
                elif Z_cam_initial is not None:
                    Z_cam = Z_cam_initial
                else:
                    Z_cam = np.linalg.norm(window_3d_pos - current_pose['position'])
                
                # Compute Z correction using reprojection
                fy = self.pnp_estimator.camera_matrix_original[1, 1]
                delta_y_cam = (error_y_px * Z_cam) / fy
                delta_cam = np.array([0.0, delta_y_cam, 0.0])
                delta_body = self.pnp_estimator.R_cam_to_body @ delta_cam
                R_drone_ned = self.pnp_estimator._euler_to_rotation_matrix(
                    current_pose['rpy'][0], current_pose['rpy'][1], current_pose['rpy'][2]
                )
                delta_ned = R_drone_ned @ delta_body
                
                z_correction = delta_ned[2]
                
                # Apply Z correction if significant
                if abs(z_correction) > z_tolerance and abs(error_y_px) > 20:  # >20px vertical error (reduced threshold)
                    # Limit step to prevent overshooting
                    z_correction = np.clip(z_correction, -0.02, 0.02)  # Smaller steps: 2cm max
                    current_pose['position'][2] += z_correction
                    print(f"  Yaw iter {yaw_iter}: Yaw={np.degrees(current_pose['rpy'][2]):.1f}, Error={np.degrees(yaw_error):.1f}, Z_corr={z_correction:+.4f}m")
                else:
                    print(f"  Yaw iter {yaw_iter}: Yaw={np.degrees(current_pose['rpy'][2]):.1f}, Error={np.degrees(yaw_error):.1f}")
            else:
                print(f"  Yaw iter {yaw_iter}: Yaw={np.degrees(current_pose['rpy'][2]):.1f}, Error={np.degrees(yaw_error):.1f}")
            
            vec_to_window = window_3d_pos - current_pose['position']
            desired_yaw = np.arctan2(vec_to_window[1], vec_to_window[0])
            yaw_error = wrap_angle(desired_yaw - current_pose['rpy'][2])
            
            rgb, _, _ = self.renderer.render(current_pose['position'], current_pose['rpy'])
            self.record_frame(rgb, pose=current_pose, annotation=f"YAW+Z_ALIGN iter={yaw_iter}")
            yaw_iter += 1
        
        if yaw_iter >= max_yaw_iterations:
            print(f"  [FAIL] Yaw alignment exceeded max iterations")
            return -1

        print(f"  [OK] Yaw aligned after {yaw_iter} iterations")
        
        # ================================================================
        # PHASE 1.5: Y+Z VERIFICATION AND CORRECTION
        # ================================================================
        print("\n[PHASE 1.5] Y+Z Verification and Fine-Tuning")
        print("  Strategy: Now that yaw is aligned, verify both lateral and vertical alignment")
        print("  (Yaw alignment might have introduced small Y errors, and Z might have overshot)")
        
        # Get fresh depth estimate from current view
        rgb_verify, _, _ = self.renderer.render(current_pose['position'], current_pose['rpy'])
        
        # Try to measure pixel height for better depth estimate
        try:
            # Quick scan to get fresh mask
            scan_waypoints_verify = self.scanner.generate_scan_trajectory(current_pose)
            scan_frames_verify = []
            for wp in scan_waypoints_verify:
                rgb_wp, _, _ = self.renderer.render(wp['position'], wp['rpy'])
                scan_frames_verify.append(rgb_wp)
            
            mask_verify, _, _, _ = self.detector.detect_window(scan_frames_verify)
            
            if mask_verify is not None and np.sum(mask_verify > 0.5) > 100:
                corners_verify = self.pnp_estimator.extract_window_corners(mask_verify)
                if corners_verify is not None:
                    Z_cam = self.estimate_depth_from_height(corners_verify)
                    print(f"    Fresh depth estimate: Z={Z_cam:.3f} splat units")
                else:
                    Z_cam = 0.315  # Fallback
                    print(f"    Using fallback depth: Z={Z_cam:.3f}")
            else:
                Z_cam = 0.315
                print(f"    Using fallback depth: Z={Z_cam:.3f}")
        except Exception as e:
            print(f"    Fresh depth estimate failed: {e}")
            Z_cam = 0.315
        
        # Iteratively correct both Y and Z
        max_yz_iterations = 12  # Increased from 8 for better convergence
        yz_pixel_threshold = 30  # Tighter! (was 50px)
        
        for yz_iter in range(max_yz_iterations):
            # Project window
            proj_pixel = self.pnp_estimator.project_window_to_pixel(window_3d_pos, current_pose)
            
            if proj_pixel is None:
                print(f"    [WARN] Cannot project window")
                break
            
            # Get errors
            img_h, img_w = rgb_verify.shape[:2]
            img_center_x = img_w / 2
            img_center_y = img_h / 2
            
            error_x_px = proj_pixel[0] - img_center_x  # Lateral (Y in NED)
            error_y_px = proj_pixel[1] - img_center_y  # Vertical (Z in NED)
            
            error_mag = np.sqrt(error_x_px**2 + error_y_px**2)
            
            print(f"    YZ iter {yz_iter}: error_x={error_x_px:+.1f}px, error_y={error_y_px:+.1f}px, mag={error_mag:.1f}px")
            
            # Check convergence
            if abs(error_x_px) < yz_pixel_threshold and abs(error_y_px) < yz_pixel_threshold:
                print(f"    [OK] Y+Z aligned! (error < {yz_pixel_threshold}px)")
                break
            
            # Compute corrections using reprojection
            fx = self.pnp_estimator.camera_matrix_original[0, 0]
            fy = self.pnp_estimator.camera_matrix_original[1, 1]
            
            # Camera frame deltas
            delta_x_cam = (error_x_px * Z_cam) / fx  # Lateral
            delta_y_cam = (error_y_px * Z_cam) / fy  # Vertical
            
            delta_cam = np.array([delta_x_cam, delta_y_cam, 0.0])
            
            # Transform to body then NED
            delta_body = self.pnp_estimator.R_cam_to_body @ delta_cam
            R_drone_ned = self.pnp_estimator._euler_to_rotation_matrix(
                current_pose['rpy'][0], current_pose['rpy'][1], current_pose['rpy'][2]
            )
            delta_ned = R_drone_ned @ delta_body
            
            # Extract Y and Z corrections
            ctrl_y = delta_ned[1]
            ctrl_z = delta_ned[2]
            
            # Limit step size (smaller than Phase 2 for precision)
            max_step = 0.03  # 3cm max per step
            ctrl_magnitude = np.sqrt(ctrl_y**2 + ctrl_z**2)
            if ctrl_magnitude > max_step:
                scale = max_step / ctrl_magnitude
                ctrl_y *= scale
                ctrl_z *= scale
            
            print(f"      Corrections: Y={ctrl_y:+.4f}m, Z={ctrl_z:+.4f}m")
            
            # Apply corrections (instantaneous)
            current_pose['position'][1] += ctrl_y
            current_pose['position'][2] += ctrl_z
            
            # Bounds check
            current_pose['position'] = self.clip_position_to_bounds(current_pose['position'])
            
            # Collision check
            if doesItCollide(current_pose['position']):
                print(f"      [WARN] Collision at corrected position, reverting")
                current_pose['position'][1] -= ctrl_y
                current_pose['position'][2] -= ctrl_z
                break
            
            # Render frame
            rgb_yz, _, _ = self.renderer.render(current_pose['position'], current_pose['rpy'])
            self.record_frame(rgb_yz, pose=current_pose, annotation=f"YZ_VERIFY iter={yz_iter}")
        
        print(f"  [OK] Y+Z verification complete")
        
        # ================================================================
        # ================================================================
        # PHASE 2: FINAL CENTERING CHECK
        # ================================================================
        print("\n[PHASE 2] Final Centering Check")
        print("  Strategy: Quick verification that alignment from Phase 1.5 is still good")
        
        # Just verify we're still centered (no corrections needed if Phase 1.5 worked)
        proj_pixel = self.pnp_estimator.project_window_to_pixel(window_3d_pos, current_pose)
        
        if proj_pixel is not None:
            rgb_final, _, _ = self.renderer.render(current_pose['position'], current_pose['rpy'])
            img_h, img_w = rgb_final.shape[:2]
            img_center_x = img_w / 2
            img_center_y = img_h / 2
            
            error_x_px = proj_pixel[0] - img_center_x
            error_y_px = proj_pixel[1] - img_center_y
            error_mag = np.sqrt(error_x_px**2 + error_y_px**2)
            
            print(f"    Final pixel error: ({error_x_px:+.1f}, {error_y_px:+.1f}), mag={error_mag:.1f}px")
            
            # Visualize
            rgb_debug = rgb_final.copy()
            cv2.drawMarker(rgb_debug, (int(img_center_x), int(img_center_y)),
                          (0, 0, 255), cv2.MARKER_CROSS, 50, 3)
            
            if proj_pixel is not None:
                px_pnp = (int(round(proj_pixel[0])), int(round(proj_pixel[1])))
                cv2.circle(rgb_debug, px_pnp, 14, (255, 0, 255), 3)
                cv2.arrowedLine(rgb_debug, (int(img_center_x), int(img_center_y)),
                              px_pnp, (255, 255, 0), 3, tipLength=0.1)
            
            self.record_frame(rgb_debug, pose=current_pose,
                            annotation=f"PHASE2_CHECK err_mag={error_mag:.0f}px")
            
            if error_mag < 100:
                print(f"  [OK] Window centered (error < 100px)")
            else:
                print(f"  [WARN] Window not perfectly centered but within tolerance")
        
        servo_converged = True
        
        print(f"\n  [OK] Visual servoing complete")
        
        # ================================================================
        
        # ================================================================
        # PHASE 3: VERIFICATION AND PASS THROUGH
        # ================================================================
        print("\n[PHASE 3] Verification and Pass Through")
        
        # Skip the long approach trajectory - we're already well-aligned!
        # Go straight to verification then pass through
        
        distance_to_window = np.linalg.norm(window_3d_pos - current_pose['position'])
        print(f"  Distance to window: {distance_to_window:.3f}m")
        print(f"  Skipping long approach - already aligned from visual servoing!")
        
        # Go directly to navigate_through_window which does verification + pass through
        return current_pose
        
    def navigate_through_window(self, current_pose, window_3d_pos, pose_history):
        """Navigate through window - VERIFY FIRST, then approach"""
        print(f"\n=== NAVIGATING THROUGH WINDOW {self.window_count + 1} ===")
        
        direction = window_3d_pos - current_pose['position']
        distance = np.linalg.norm(direction)
        
        print(f"  Current: {current_pose['position']}")
        print(f"  Window: {window_3d_pos}")
        print(f"  Distance: {distance:.3f}m")
        
        # ================================================================
        # ITERATIVE ALIGN-VERIFY LOOP
        # ================================================================
        print(f"\n  ========================================")
        print(f"  ITERATIVE ALIGN-VERIFY LOOP")
        print(f"  ========================================")
        print(f"  Strategy: Repeatedly verify and align until error is consistently low")
        
        max_align_verify_cycles = 5
        align_threshold = 50  # Pixels - must be below this to proceed to approach
        
        alignment_good = False
        
        for cycle in range(max_align_verify_cycles):
            print(f"\n  === Cycle {cycle + 1}/{max_align_verify_cycles} ===")
            
            # STEP 1: VERIFY (scan and detect)
            print(f"  Verifying...")
            verified_window, _, verified_center, verified_corners = self.scan_for_window(
                current_pose, pose_history, scan_type=f'verify_cycle{cycle}'
            )
            
            if verified_window is None:
                print(f"  [ERROR] Verification failed - cannot see window!")
                if cycle == 0:
                    return -1  # First verification failed - abort
                else:
                    print(f"  [WARN] Lost window during alignment - using last known position")
                    break  # Use last good position
            
            # Update window position with latest estimate
            window_3d_pos = verified_window
            
            # Check position consistency
            if cycle > 0:
                pos_drift = np.linalg.norm(verified_window - window_3d_pos)
                print(f"  Position drift from last cycle: {pos_drift:.3f} splat units")
            
            # STEP 2: CHECK ALIGNMENT
            print(f"  Checking alignment...")
            proj_pixel = self.pnp_estimator.project_window_to_pixel(window_3d_pos, current_pose)
            
            if proj_pixel is None:
                print(f"  [WARN] Cannot project window")
                continue
            
            rgb_check, _, _ = self.renderer.render(current_pose['position'], current_pose['rpy'])
            img_h, img_w = rgb_check.shape[:2]
            img_center_x = img_w / 2
            img_center_y = img_h / 2
            
            error_x_px = proj_pixel[0] - img_center_x
            error_y_px = proj_pixel[1] - img_center_y
            error_mag = np.sqrt(error_x_px**2 + error_y_px**2)
            
            print(f"  Alignment error: ({error_x_px:+.1f}, {error_y_px:+.1f})px, mag={error_mag:.1f}px")
            
            # Visualize
            rgb_annotated = rgb_check.copy()
            cv2.drawMarker(rgb_annotated, (int(img_center_x), int(img_center_y)),
                          (0, 0, 255), cv2.MARKER_CROSS, 50, 3)
            
            if proj_pixel is not None:
                px_proj = (int(round(proj_pixel[0])), int(round(proj_pixel[1])))
                cv2.circle(rgb_annotated, px_proj, 20, (255, 0, 255), 3)
                cv2.putText(rgb_annotated, f'Cycle {cycle+1}', (px_proj[0]+25, px_proj[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                cv2.arrowedLine(rgb_annotated, (int(img_center_x), int(img_center_y)),
                              px_proj, (255, 255, 0), 3, tipLength=0.1)
                
                # Draw verified corners if available
                if verified_corners is not None:
                    corners_int = verified_corners.astype(np.int32)
                    cv2.polylines(rgb_annotated, [corners_int], True, (255, 255, 0), 2)
            
            self.record_frame(rgb_annotated, pose=current_pose,
                            annotation=f"ALIGN_VERIFY_C{cycle}_ERR={error_mag:.0f}px")
            
            # STEP 3: CHECK IF ALIGNED
            if error_mag < align_threshold:
                print(f"  [OK] Alignment good! (error {error_mag:.1f}px < {align_threshold}px)")
                alignment_good = True
                break
            
            # STEP 4: ALIGN (correct position)
            print(f"  Aligning to reduce error...")
            
            # Estimate depth
            if verified_corners is not None:
                try:
                    Z_cam = self.estimate_depth_from_height(verified_corners)
                    print(f"    Depth from corners: Z={Z_cam:.3f}")
                except:
                    Z_cam = 0.3
                    print(f"    Using default depth: Z={Z_cam:.3f}")
            else:
                Z_cam = 0.3
            
            # Compute corrections
            fx = self.pnp_estimator.camera_matrix_original[0, 0]
            fy = self.pnp_estimator.camera_matrix_original[1, 1]
            
            delta_x_cam = (error_x_px * Z_cam) / fx
            delta_y_cam = (error_y_px * Z_cam) / fy
            delta_cam = np.array([delta_x_cam, delta_y_cam, 0.0])
            
            # Transform to NED
            delta_body = self.pnp_estimator.R_cam_to_body @ delta_cam
            R_drone_ned = self.pnp_estimator._euler_to_rotation_matrix(
                current_pose['rpy'][0], current_pose['rpy'][1], current_pose['rpy'][2]
            )
            delta_ned = R_drone_ned @ delta_body
            
            ctrl_y = delta_ned[1]
            ctrl_z = delta_ned[2]
            
            # Limit step size
            max_step = 0.05
            ctrl_magnitude = np.sqrt(ctrl_y**2 + ctrl_z**2)
            if ctrl_magnitude > max_step:
                scale = max_step / ctrl_magnitude
                ctrl_y *= scale
                ctrl_z *= scale
            
            print(f"    Corrections: Y={ctrl_y:+.4f}, Z={ctrl_z:+.4f}")
            
            # Apply
            new_pos = current_pose['position'].copy()
            new_pos[1] += ctrl_y
            new_pos[2] += ctrl_z
            new_pos = self.clip_position_to_bounds(new_pos)
            
            # Collision check
            if doesItCollide(new_pos):
                print(f"    [ERROR] Correction would cause collision!")
                break
            
            current_pose['position'] = new_pos
            print(f"    [OK] Moved to {current_pose['position']}")
        
        if not alignment_good:
            print(f"\n  [WARN] Alignment did not converge after {max_align_verify_cycles} cycles")
            print(f"  Final error: {error_mag:.1f}px")
            if error_mag > 100:
                print(f"  [ERROR] Error too large to safely approach!")
                return -1
        
        print(f"  [OK] Iterative align-verify complete!")
        
        # ================================================================
        # FINAL PRE-APPROACH CHECK
        # ================================================================
        print(f"\n  Pre-approach alignment check...")
        proj_pixel_pre = self.pnp_estimator.project_window_to_pixel(window_3d_pos, current_pose)
        
        if proj_pixel_pre is not None:
            rgb_check, _, _ = self.renderer.render(current_pose['position'], current_pose['rpy'])
            img_h, img_w = rgb_check.shape[:2]
            img_center_x = img_w / 2
            img_center_y = img_h / 2
            
            error_x_pre = proj_pixel_pre[0] - img_center_x
            error_y_pre = proj_pixel_pre[1] - img_center_y
            error_mag_pre = np.sqrt(error_x_pre**2 + error_y_pre**2)
            
            print(f"    Pixel error: ({error_x_pre:+.1f}, {error_y_pre:+.1f}), mag={error_mag_pre:.1f}px")
            
            if error_mag_pre > 100:
                print(f"    [ERROR] Alignment not good enough! (error={error_mag_pre:.1f}px > 100px)")
                print(f"    Cannot safely approach - must re-align first")
                return -1
            else:
                print(f"    [OK] Alignment verified (error < 100px)")
        
        # ================================================================
        # STEP 3: APPROACH (Now that we've verified!)
        # ================================================================
        approach_distance = 0.2  # Get this close before pass-through
        
        if distance > approach_distance:
            # CRITICAL: Move forward along camera axis, NOT toward window estimate!
            # The window estimate might be slightly off, but our camera is pointed at it
            # So moving forward is safer than moving toward a potentially inaccurate 3D position
            
            R_drone_ned = self.pnp_estimator._euler_to_rotation_matrix(
                current_pose['rpy'][0], current_pose['rpy'][1], current_pose['rpy'][2]
            )
            forward_direction = R_drone_ned @ np.array([1, 0, 0])  # Body X = forward in NED
            
            # How far forward to move
            move_distance = distance - approach_distance
            approach_point = current_pose['position'] + forward_direction * move_distance
            
            print(f"  Moving forward {move_distance:.3f}m along camera axis")
            print(f"  Approach target: {approach_point}")
            
            if not self.is_position_in_bounds(approach_point):
                print(f"  [ERROR] Approach point outside map bounds: {approach_point}")
                approach_point = self.clip_position_to_bounds(approach_point)
                print(f"  [WARN] Clipped to: {approach_point}")
            
            if doesItCollide(approach_point):
                print(f"  [ERROR] Approach point collides")
                return -1
            
            result = goToWaypoint(
                current_pose, approach_point,
                velocity=0.03,  # SLOW! (reduced from 0.05)
                pose_history=pose_history,
                action=f'APPROACH_W{self.window_count}',
                lock_roll_pitch=True,
                navigator=self
            )
            
            if result == -1:
                return -1
            
            current_pose = result
            print(f"  Approached")
        
        # ================================================================
        # STEP 4: PASS THROUGH
        # ================================================================
        print(f"\n  Passing through window...")
        through_distance = 0.5
        direction = window_3d_pos - current_pose['position']
        if np.linalg.norm(direction) > 0.01:
            direction_norm = direction / np.linalg.norm(direction)
        else:
            # Already at window, use forward direction
            R_drone = self.pnp_estimator._euler_to_rotation_matrix(
                current_pose['rpy'][0], current_pose['rpy'][1], current_pose['rpy'][2]
            )
            direction_norm = R_drone @ np.array([1, 0, 0])
        
        through_point = window_3d_pos + direction_norm * through_distance
        
        if not self.is_position_in_bounds(through_point):
            print(f"  [ERROR] Through point outside map bounds: {through_point}")
            through_point = self.clip_position_to_bounds(through_point)
            print(f"  [WARN] Clipped to: {through_point}")
        
        if doesItCollide(through_point):
            print(f"  [ERROR] Through point collides")
            return -1
        
        result = goToWaypoint(
            current_pose, through_point,
            velocity=0.05,
            pose_history=pose_history,
            action=f'PASS_THROUGH_W{self.window_count}',
            lock_roll_pitch=True,
            navigator=self
        )
        
        if result == -1:
            return -1
        
        current_pose = result
        print(f"  Passed through")
        
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
        """Save frame with pose overlay"""
        # NOTE: Frame deletion is handled in main() at startup, NOT here!
        # This prevents accidental deletion during skill execution
        
        if frame_id is None:
            frame_id = len(self.video_frames)
        
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
            print(f"\nSaved {num_frames} frames to ./log/frames/")


def main(renderer):
    os.makedirs('./log', exist_ok=True)
    os.makedirs('./log/frames', exist_ok=True)
    
    import glob
    # Clean up old PNG files
    for f in glob.glob('./log/*.png'):
        try:
            os.remove(f)
        except:
            pass
    
    # CRITICAL: Clean up old frame files!
    for f in glob.glob('./log/frames/*.png'):
        try:
            os.remove(f)
        except:
            pass
    
    print(f"  [OK] Cleaned up old frames")
    
    print("\n" + "="*70)
    print("DRONE RACING - 6-SKILL ARCHITECTURE")
    print("="*70)
    print("Skills: SCAN  FIX_YAW  ALIGN-VERIFY LOOP  APPROACH  RECENTER")
    print("="*70 + "\n")

    # Reset collision checker
    import collisionChecker
    collisionChecker._default_checker = None
    
    navigator = WindowNavigator(renderer, device='cuda')
    
    # Import skills module
    from skills import NavigationSkills
    skills = NavigationSkills(navigator)
    
    currentPose = {
        'position': np.array([0.0, 0.0, 0.0]),
        'rpy': np.radians([0.0, 0.0, 0.0])
    }
    
    # Use navigator's pose_history
    navigator.pose_history.append({
        'step': 0,
        'action': 'INITIAL',
        'position': currentPose['position'].copy(),
        'rpy_deg': np.degrees(currentPose['rpy']),
        'rpy_rad': currentPose['rpy'].copy()
    })
    
    print(f"Initial position: {currentPose['position']}")
    print(f"Checking collision at origin...")
    
    if doesItCollide(currentPose['position']):
        print(f"[ERROR] Origin collides! Cannot start navigation.")
        return -1
    
    print(f"[OK] Origin is collision-free")
    
    rgb, _, _ = renderer.render(currentPose['position'], currentPose['rpy'])
    navigator.record_frame(rgb, pose=currentPose, annotation="START")
    
    print("\n" + "="*70)
    print("NAVIGATION LOOP")
    print("="*70)
    
    max_windows = 3
    
    for window_num in range(max_windows):
        print(f"\n{'='*70}")
        print(f"WINDOW {window_num + 1} / {max_windows}")
        print(f"{'='*70}")
        print(f"Flow: SCAN  [FIX_YAW if >5]  ALIGN  VERIFY  APPROACH  RECENTER")
        
        # =================================================================
        # SKILL 1: SCAN
        # =================================================================
        window_3d_pos, corners_2d, scan_data = skills.scan(currentPose, scan_type='initial')
        
        if window_3d_pos is None:
            print(f"\n[ABORT] Could not detect window {window_num + 1}")
            break
        
        # =================================================================
        # SKILL 2: FIX_YAW (Conditional - only if needed)
        # =================================================================
        # Check if yaw alignment is needed
        vec_to_window = window_3d_pos - currentPose['position']
        desired_yaw = np.arctan2(vec_to_window[1], vec_to_window[0])
        current_yaw = currentPose['rpy'][2]
        yaw_error = wrap_angle(desired_yaw - current_yaw)
        yaw_error_deg = np.degrees(abs(yaw_error))
        
        # ALWAYS store initial yaw error for VERIFY guidance (even if we skip FIX_YAW)
        skills.yaw_error_initial = yaw_error
        
        print(f"\nYaw check: current={np.degrees(current_yaw):.1f}, "
              f"desired={np.degrees(desired_yaw):.1f}, error={yaw_error_deg:.1f}")
        print(f"  Yaw hint for VERIFY: {np.degrees(yaw_error):.1f} ({'RIGHT' if yaw_error > 0 else 'LEFT'})")
        
        if yaw_error_deg > 5.0:
            print(f"   Yaw error {yaw_error_deg:.1f} > 5 - running FIX_YAW")
            result = skills.fix_yaw(currentPose, window_3d_pos)
            
            if result == -1:
                print(f"\n[ABORT] Yaw alignment failed")
                break
            
            currentPose = result
        else:
            print(f"   Yaw error {yaw_error_deg:.1f} < 5 - skipping FIX_YAW ")
            print(f"  Already well aligned!")
        
        # =================================================================
        # SKILLS 3-4: ALIGN-VERIFY LOOP (Conditional)
        # =================================================================
        print(f"\n{'='*70}")
        print(f"[ALIGN-VERIFY CHECK]")
        print(f"{'='*70}")
        
        # NOW check alignment (after yaw is done!)
        proj_pixel = navigator.pnp_estimator.project_window_to_pixel(window_3d_pos, currentPose)
        
        if proj_pixel is not None:
            img_h, img_w = renderer.image_height, renderer.image_width
            error_x = proj_pixel[0] - img_w / 2
            error_y = proj_pixel[1] - img_h / 2
            error_mag = np.sqrt(error_x**2 + error_y**2)
            
            print(f"Post-yaw-alignment error: {error_mag:.1f}px")
            
            if error_mag < 25:  # Tighter threshold - approach only if very well aligned
                print(f"   Alignment excellent ({error_mag:.1f}px < 25px)")
                print(f"   Skipping ALIGN-VERIFY, going straight to APPROACH!")
            else:
                print(f"   Alignment needs refinement ({error_mag:.1f}px > 25px)")
                print(f"   Running ALIGN-VERIFY cycles until converged")
                
                # ITERATIVE ALIGN-VERIFY LOOP (max 3 cycles)
                max_cycles = 3
                for cycle_num in range(max_cycles):
                    print(f"\n  --- ALIGN-VERIFY Cycle {cycle_num + 1}/{max_cycles} ---")
                    
                    # SKILL 3: ALIGN (adjust Y/Z position)
                    currentPose, error_before_verify = skills.align(currentPose, window_3d_pos, corners_2d)
                    
                    # SKILL 4: VERIFY (re-scan to confirm)
                    verified_window, corners_2d, alignment_good, error_mag = skills.verify(
                        currentPose, window_3d_pos, cycle_num=cycle_num
                    )
                    
                    if verified_window is None:
                        print(f"  [WARN] Verification failed (likely detected different window)")
                        print(f"  [RETRY] Moving backward slightly to reject second window...")
                        
                        # Move backward in NED coordinates (negative X)
                        step_size = 0.02
                        new_pos = currentPose['position'].copy()
                        new_pos[0] -= step_size  # -X in NED = backward/south
                        new_pos = navigator.clip_position_to_bounds(new_pos)
                        
                        print(f"  Moving -X (backward) by {step_size:.3f} units")
                        print(f"  From: {currentPose['position']}")
                        print(f"  To: {new_pos}")
                        
                        currentPose['position'] = new_pos
                        
                        rgb_retry, _, _ = renderer.render(currentPose['position'], currentPose['rpy'])
                        navigator.record_frame(rgb_retry, pose=currentPose, 
                                             annotation="VERIFY_RETRY_BACK")
                        
                        # RETRY VERIFY after moving backward
                        print(f"  [RETRY] Re-verifying after backward step...")
                        verified_window, corners_2d, alignment_good, error_mag = skills.verify(
                            currentPose, window_3d_pos, cycle_num=cycle_num
                        )
                        
                        if verified_window is None:
                            print(f"  [WARN] Second verify also failed")
                            print(f"  [OK] Continuing with original scan position")
                            break
                        else:
                            print(f"  [SUCCESS] Second verify succeeded!")
                            window_3d_pos = verified_window
                    else:
                        # Verify succeeded
                        window_3d_pos = verified_window
                    
                    # Check convergence
                    if alignment_good:
                        print(f"  [CONVERGED] Excellent alignment! (error={error_mag:.1f}px < 50px)")
                        break
                    elif error_mag < 20:
                        print(f"  [GOOD ENOUGH] Ready to approach (error={error_mag:.1f}px < 20px)")
                        break
                    else:
                        print(f"  [CONTINUE] Need more refinement (error={error_mag:.1f}px)")
                        if cycle_num < max_cycles - 1:
                            print(f"   Running another ALIGN-VERIFY cycle")
                        else:
                            print(f"  [STOP] Reached max cycles, proceeding anyway")
        else:
            print(f"  [WARN] Cannot project window - skipping ALIGN-VERIFY")
        
        # =================================================================
        # SKILL 5: APPROACH
        # =================================================================
        result = skills.approach(currentPose, window_3d_pos)
        
        if result == -1:
            print(f"\n[ABORT] Approach failed - stopping navigation")
            break
        
        currentPose = result
        navigator.window_count += 1
        
        # =================================================================
        # SKILL 6: RECENTER
        # =================================================================
        currentPose = skills.recenter(currentPose)
        
        print(f"\n{'='*70}")
        print(f"[OK] Window {window_num + 1} complete!")
        print(f"{'='*70}")
        
        # Ready for next window - loop will call SCAN again
    
    # Save results
    with open('./log/pose_history.json', 'w') as f:
        json.dump(navigator.pose_history, f, indent=2, default=str)
    
    navigator.save_frames_summary()
    
    print("\n" + "="*70)
    print("NAVIGATION COMPLETE")
    print(f"  Windows passed: {navigator.window_count}")
    print("="*70 + "\n")


if __name__ == "__main__":
    config_path = "../data/P5_colmap_splat/P5_colmap/splatfacto/2025-11-17_130359/config.yml"
    json_path = "../data/render_settings/render_settings.json"

    renderer = SplatRenderer(config_path, json_path)
    main(renderer)