"""
Drone Racing Navigation System - WITH PnP INTEGRATION
[DONE] PnP-based window pose estimation
[DONE] Yaw changes use goToWaypoint (not instant)
[DONE] Coordinate transformation fixed
[DONE] Frame rendering during visual servoing
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
        
        # PnP scale factor - tune this to match real world scale
        # If projected box is too large (outside frame), INCREASE this value (pushes window further)
        # If projected box is too small, DECREASE this value (brings window closer)
        self.PNP_SCALE_FACTOR = 2.0  # Start with 2.0 (makes window 2x further than PnP reports)
        
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
    
    
    def _check_corner_consistency(self, detected_corners_2d, window_3d_pos, window_rpy_ned, camera_pose, threshold_pixels=200):
        """
        Check if detected corners match the projected corners from 3D estimate.
        This prevents window-jumping when detector locks onto a different window.
        
        Args:
            detected_corners_2d: (4,2) array of detected corners
            window_3d_pos: (3,) current window position estimate
            window_rpy_ned: (3,) current window orientation
            camera_pose: Current camera pose dict
            threshold_pixels: Maximum average distance for match
            
        Returns:
            bool: True if corners match (same window), False if different window
        """
        if detected_corners_2d is None:
            return False
        
        # Project current 3D estimate to get expected corners
        w = self.pnp_estimator.window_width / 2
        h = self.pnp_estimator.window_height / 2
        R_window = self.pnp_estimator._euler_to_rotation_matrix(
            window_rpy_ned[0], window_rpy_ned[1], window_rpy_ned[2]
        )
        corners_local = np.array([
            [-w, -h, 0], [w, -h, 0], [w, h, 0], [-w, h, 0]
        ])
        window_corners_3d = np.array([
            window_3d_pos + R_window @ corner for corner in corners_local
        ])
        
        # Project to image
        projected_corners = []
        for corner_3d in window_corners_3d:
            corner_2d = self.pnp_estimator.project_window_to_pixel(corner_3d, camera_pose)
            if corner_2d is not None:
                projected_corners.append(corner_2d)
        
        if len(projected_corners) != 4:
            print(f"    Corner check: Can't project all 4 corners")
            return False
        
        # Calculate average pixel distance
        corner_distances = [
            np.linalg.norm(detected_corners_2d[i] - projected_corners[i])
            for i in range(4)
        ]
        avg_dist = np.mean(corner_distances)
        max_dist = np.max(corner_distances)
        
        is_match = avg_dist < threshold_pixels
        
        if is_match:
            print(f"    [OK] Corner match: avg={avg_dist:.1f}px, max={max_dist:.1f}px (SAME window)")
        else:
            print(f"    [FAIL] Corner mismatch: avg={avg_dist:.1f}px, max={max_dist:.1f}px (DIFFERENT window!)")
        
        return is_match
    
    def scan_for_window(self, current_pose, pose_history, scan_type='initial', target_pixel=None):
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
            
            print(f"    Frame {i+1}/{len(scan_waypoints)}")
        
        # Detect window
        print("\n  Detecting window...")
        mask, center_2d, confidence, debug_info = self.detector.detect_window(scan_frames)

        # --- Extra debug: label and visualize all contours with indices, areas, centroids ---
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
                        # draw contour in light blue
                        cv2.drawContours(overlay, [c.astype(np.int32)], -1, (200, 200, 255), -1)
                        # label index and area
                        cv2.putText(ref_img, f"{idx_c}", (cx+5, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
                        cv2.putText(ref_img, f"{idx_c}", (cx+5, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                        cv2.putText(ref_img, f"A:{int(area)}", (cx+5, cy+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

                    vis_labeled = cv2.addWeighted(ref_img, 0.7, overlay, 0.3, 0)
                    labeled_path = f"./log/detection_{scan_label}_labeled.png"
                    os.makedirs('./log', exist_ok=True)
                    cv2.imwrite(labeled_path, cv2.cvtColor(vis_labeled, cv2.COLOR_RGB2BGR))
                    print(f"  [DEBUG] Saved labeled contour visualization: {labeled_path}")
                    # Create a mask visualization with labels (indexes) on the thresholded mask
                    try:
                        mask_viz = cv2.cvtColor(mask_uint8_dbg, cv2.COLOR_GRAY2BGR)
                        # draw contour outlines and index labels directly on mask_viz
                        for idx_c, c in enumerate(contours_dbg):
                            area = cv2.contourArea(c)
                            M = cv2.moments(c)
                            if M['m00'] == 0:
                                cx, cy = -1, -1
                            else:
                                cx = int(M['m10'] / M['m00'])
                                cy = int(M['m01'] / M['m00'])
                            # outline contour in cyan
                            cv2.drawContours(mask_viz, [c.astype(np.int32)], -1, (255, 200, 200), 2)
                            # draw centroid marker
                            if cx >= 0 and cy >= 0:
                                cv2.circle(mask_viz, (cx, cy), 6, (0, 0, 255), -1)
                                # put index and area text
                                cv2.putText(mask_viz, f"{idx_c}", (cx+8, cy+6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                                cv2.putText(mask_viz, f"A:{int(area)}", (cx+8, cy+28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                        mask_labeled_path = f"./log/detection_{scan_label}_mask_labeled.png"
                        cv2.imwrite(mask_labeled_path, cv2.cvtColor(mask_viz, cv2.COLOR_RGB2BGR))
                        print(f"  [DEBUG] Saved mask-labeled visualization: {mask_labeled_path}")
                    except Exception as e:
                        print(f"  [WARN] Failed to save mask-labeled visualization: {e}")
                    # print concise contour stats
                    print("  [DEBUG] Contour stats (idx, area, centroid):")
                    for s in stats:
                        print(f"    - {s[0]}: area={s[1]:.0f}, centroid={s[2]}")
        except Exception as e:
            print(f"  [WARN] Failed to create labeled contour visualization: {e}")

        # If caller provided an expected pixel location (from previous PnP), try to prefer
        # the contour closest to that projected pixel. This avoids the detector locking
        # onto a different window (e.g., upper hole of window 2) after yaw.
        if target_pixel is not None and mask is not None:
            try:
                mask_uint8 = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) > 0:
                    # Compute centroid for each contour and pick nearest to target
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

                    if best_idx is not None and best_dist < 500:  # pixel radius threshold
                        chosen = contours[best_idx]
                        chosen_centroid = centroids[best_idx]
                        print(f"  [INFO] Selecting contour closest to projected pixel {target_pixel}: dist={best_dist:.1f}px")
                        # Create new mask containing only the chosen contour
                        new_mask = np.zeros_like(mask_uint8)
                        cv2.drawContours(new_mask, [chosen], -1, 255, -1)
                        mask = (new_mask.astype(np.float32) / 255.0)
                        center_2d = (chosen_centroid[0], chosen_centroid[1])
                        # Update debug_info to reflect selection (if possible)
                        debug_info = debug_info if debug_info is not None else {}
                        debug_info['preferred_contour_selected'] = True
                    else:
                        print(f"  [INFO] No contour close enough to projected pixel (best_dist={best_dist:.1f}) - using default mask")
            except Exception as e:
                print(f"  [WARN] Targeted contour selection failed: {e}")
        
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
        
        print(f"  Window at pixel {center_2d} (conf: {confidence:.3f})")
        
        # PnP POSE ESTIMATION
        print("\n  Estimating pose with PnP...")
        
        # CRITICAL: Use first frame as reference since flow computation uses first-last
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
        
        # Transform to NED with SCALE FACTOR
        window_pos_ned, window_rpy_ned = self.pnp_estimator.transform_to_ned(
            tvec_cam * self.PNP_SCALE_FACTOR,  # Scale the translation vector
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
                return None, scan_frames, None
        
        # CRITICAL: Check if window position is within map bounds
        if not self.is_position_in_bounds(window_pos_ned):
            print(f"  [ERROR] Window position outside map bounds!")
            print(f"    Position: X={window_pos_ned[0]:.2f}, Y={window_pos_ned[1]:.2f}, Z={window_pos_ned[2]:.2f}")
            print(f"    Limits: X=+/-{self.MAP_X_LIMIT}, Y=+/-{self.MAP_Y_LIMIT}, Z=+/-{self.MAP_Z_LIMIT}")
            return None, scan_frames, None
        
        # Ensure frames are saved with pose overlay through record_frame
        for idx, (frame, pose) in enumerate(zip(scan_frames, scan_poses)):
            self.record_frame(frame, pose=pose, annotation=f"SCAN {scan_type}")
            print(f"  [OK] Saved scan frame {idx+1}/{len(scan_frames)}")
        
        return window_pos_ned, scan_frames, center_2d, corners_2d
    
    def _draw_window_center_crosshair(self, rgb_image, pixel_coords, corners_2d=None):
        """
        Draw a crosshair at the projected window center and optionally the PnP corners
        
        Args:
            rgb_image: RGB image to draw on
            pixel_coords: (x, y) pixel coordinates of window center
            corners_2d: Optional (4, 2) array of PnP corner coordinates
            
        Returns:
            Modified RGB image
        """
        img = rgb_image.copy()
        x, y = int(pixel_coords[0]), int(pixel_coords[1])
        
        # Draw crosshair (red)
        crosshair_size = 30
        thickness = 3
        color = (255, 0, 0)  # Red
        
        # Horizontal line
        cv2.line(img, (x - crosshair_size, y), (x + crosshair_size, y), color, thickness)
        # Vertical line
        cv2.line(img, (x, y - crosshair_size), (x, y + crosshair_size), color, thickness)
        
        # Circle at center
        cv2.circle(img, (x, y), 8, color, thickness)
        
        # Draw PnP corners and box if provided
        if corners_2d is not None and len(corners_2d) == 4:
            # Draw corners (yellow circles)
            for i, corner in enumerate(corners_2d):
                cx, cy = int(corner[0]), int(corner[1])
                cv2.circle(img, (cx, cy), 8, (255, 255, 0), -1)  # Yellow filled circle
                # Label corner
                cv2.putText(img, str(i), (cx + 10, cy + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Draw bounding box (yellow lines)
            corners_int = corners_2d.astype(np.int32)
            cv2.polylines(img, [corners_int], True, (255, 255, 0), 2)
        
        # Draw image center for reference (green)
        img_h, img_w = img.shape[:2]
        img_center_x, img_center_y = img_w // 2, img_h // 2
        cv2.circle(img, (img_center_x, img_center_y), 5, (0, 255, 0), 2)
        
        # Add text showing pixel error
        pixel_error = np.linalg.norm(np.array([x, y]) - np.array([img_center_x, img_center_y]))
        cv2.putText(img, f"Window center: ({x}, {y})", (x + 40, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(img, f"Pixel error: {pixel_error:.1f}px", (x + 40, y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return img
    
    def visual_servo_to_window(self, current_pose, window_3d_pos, window_rpy_ned, pose_history, initial_corners_2d=None, scan_frames=None):
        """
        HYBRID Visual Servoing:
        1. PnP-based yaw alignment (your robust approach)
        2. P3-style visual servoing for lateral/vertical (proven simple approach)
        3. Move forward when centered
        
        Args:
            current_pose: Current drone pose
            window_3d_pos: Window 3D position (NED)
            window_rpy_ned: Window orientation (NED)
            pose_history: Pose history list
            initial_corners_2d: Initial PnP corners from detection (4, 2) array
            scan_frames: Scan frames from detection phase (for template matching)
            
        Returns:
            final_pose: Pose when reached window or -1 if failed
        """
        print("\n" + "="*70)
        print("HYBRID VISUAL SERVOING (PnP Yaw + P3 Lateral)")
        print("="*70)
        
        # Calculate initial yaw error
        vec_to_window = window_3d_pos - current_pose['position']
        desired_yaw = np.arctan2(vec_to_window[1], vec_to_window[0])  # atan2(East, North)
        current_yaw = current_pose['rpy'][2]
        yaw_error = wrap_angle(desired_yaw - current_yaw)

        # ================================================================
        # PHASE 1: YAW ALIGNMENT (Your existing approach - WORKS!)
        # ================================================================
        print("\n[PHASE 1] PnP-Based Yaw Alignment")
        yaw_tolerance = np.radians(1.0)  # 1 degree tolerance
        yaw_step = np.radians(1.0)  # Increment yaw by 1 degree per iteration
        
        yaw_iter = 0
        max_yaw_iterations = 30  # Safety limit
        
        while abs(yaw_error) > yaw_tolerance and yaw_iter < max_yaw_iterations:
            # Incrementally adjust yaw
            step_yaw = np.clip(yaw_error, -yaw_step, yaw_step)
            current_pose['rpy'][2] += step_yaw  # Adjust yaw only
            
            # NO PNP UPDATES DURING ROTATION!
            # Just rotate using initial window estimate
            
            # Calculate yaw error
            vec_to_window = window_3d_pos - current_pose['position']
            desired_yaw = np.arctan2(vec_to_window[1], vec_to_window[0])
            yaw_error = wrap_angle(desired_yaw - current_pose['rpy'][2])

            print(f"  Yaw iter {yaw_iter}: Current={np.degrees(current_pose['rpy'][2]):.1f}°, Target={np.degrees(desired_yaw):.1f}°, Error={np.degrees(yaw_error):.1f}°")
            
            # Render frame (NO corner visualization during rotation)
            rgb, _, _ = self.renderer.render(current_pose['position'], current_pose['rpy'])
            self.record_frame(rgb, pose=current_pose, annotation=f"YAW_ALIGN iter={yaw_iter}")
            yaw_iter += 1
        
        if yaw_iter >= max_yaw_iterations:
            print(f"  [FAIL] Yaw alignment exceeded max iterations")
            return -1

        print(f"  [OK] Yaw aligned after {yaw_iter} iterations")
        
        # ================================================================
        # PHASE 1.5: SKIP VERIFICATION - USE INITIAL WINDOW ESTIMATE
        # ================================================================
        print("\n[PHASE 1.5] Using Initial Window Estimate (Skipping Re-detection)")
        print("  Window is already detected and yaw-aligned. Proceeding to centering.")
        
        # Update window position with yaw-aligned pose for better tracking
        # The window 3D position is still valid - we just rotated to face it
        
        # ================================================================
        # PHASE 2: PnP-BASED WINDOW CENTERING (Re-detect each iteration)
        # ================================================================
        print("\n[PHASE 2] PnP-Based Window Centering")
        print("  Strategy: Scan + Detect + PnP on each iteration for robust tracking")
        
        # Control parameters
        lateral_pixel_threshold = 100  # Pixels tolerance for centering (relaxed for PnP)
        max_servo_iterations = 8
        
        servo_converged = False
        
        for servo_iter in range(max_servo_iterations):
            print(f"\n  Visual Servo Iteration {servo_iter + 1}/{max_servo_iterations}")
            print(f"    Current position: {current_pose['position']}")
            print(f"    Current yaw: {np.degrees(current_pose['rpy'][2]):.1f}°")
            
            # PROJECT window to current view to get expected pixel location
            proj_pixel = self.pnp_estimator.project_window_to_pixel(window_3d_pos, current_pose)
            
            if proj_pixel is None:
                print(f"    [FAIL] Cannot project window to image plane!")
                break
            
            print(f"    Expected window center at pixel: ({proj_pixel[0]:.1f}, {proj_pixel[1]:.1f})")
            
            # Render current view
            rgb, _, _ = self.renderer.render(current_pose['position'], current_pose['rpy'])
            img_h, img_w = rgb.shape[:2]
            img_center_x = img_w / 2
            img_center_y = img_h / 2
            
            # Compute pixel error from projected location
            error_x_px = proj_pixel[0] - img_center_x
            error_y_px = proj_pixel[1] - img_center_y
            
            print(f"    Image center: ({img_center_x:.0f}, {img_center_y:.0f})")
            print(f"    Pixel error: ({error_x_px:.1f}, {error_y_px:.1f})")
            error_mag = np.sqrt(error_x_px**2 + error_y_px**2)
            print(f"    Error magnitude: {error_mag:.1f} pixels")
            
            # Check if centered
            if abs(error_x_px) < lateral_pixel_threshold and abs(error_y_px) < lateral_pixel_threshold:
                print(f"  [OK] CENTERED! Window aligned within {lateral_pixel_threshold} pixels")
                servo_converged = True
                break
            
            # Estimate distance to window (used for reprojection)
            distance_to_window = np.linalg.norm(window_3d_pos - current_pose['position'])
            print(f"    Distance to window: {distance_to_window:.3f}m")

            # --- Reprojection-based pixel->meter mapping (physically consistent) ---
            # Try to compute depth Z from measured pixel height if available
            H_real = 0.067  # Known real window height in splat units
            h_px_measured = None

            # --- Per-iteration re-detection to refresh pixel height ---
            try:
                scan_waypoints = self.scanner.generate_scan_trajectory(current_pose)
                scan_frames_iter = []
                for idx, wp in enumerate(scan_waypoints):
                    rgb_wp, _, _ = self.renderer.render(wp['position'], wp['rpy'])
                    scan_frames_iter.append(rgb_wp)

                mask_iter, center_iter, conf_iter, debug_iter = self.detector.detect_window(scan_frames_iter)
                if mask_iter is not None and np.sum(mask_iter > 0.5) > 50:
                    # extract corners from mask (original resolution)
                    try:
                        corners_iter = self.pnp_estimator.extract_window_corners(mask_iter)
                        if corners_iter is not None:
                            top_y = 0.5 * (corners_iter[2, 1] + corners_iter[3, 1])
                            bot_y = 0.5 * (corners_iter[0, 1] + corners_iter[1, 1])
                            h_px_measured = abs(bot_y - top_y)
                            print(f"    Per-iter measured h_px: {h_px_measured:.1f}px (from detector)")
                    except Exception:
                        h_px_measured = None
                else:
                    # fallback to initial corners passed from detection
                    if initial_corners_2d is not None:
                        try:
                            top_y = 0.5 * (initial_corners_2d[2, 1] + initial_corners_2d[3, 1])
                            bot_y = 0.5 * (initial_corners_2d[0, 1] + initial_corners_2d[1, 1])
                            h_px_measured = abs(bot_y - top_y)
                        except Exception:
                            h_px_measured = None
            except Exception as e:
                print(f"    [WARN] Re-detection failed: {e}")
                if initial_corners_2d is not None:
                    try:
                        top_y = 0.5 * (initial_corners_2d[2, 1] + initial_corners_2d[3, 1])
                        bot_y = 0.5 * (initial_corners_2d[0, 1] + initial_corners_2d[1, 1])
                        h_px_measured = abs(bot_y - top_y)
                    except Exception:
                        h_px_measured = None

            cam_vec = self.pnp_estimator.get_camera_vector(window_3d_pos, current_pose)

            fx = self.pnp_estimator.camera_matrix_original[0, 0]
            fy = self.pnp_estimator.camera_matrix_original[1, 1]

            if h_px_measured is not None and h_px_measured > 5:
                # Compute Z from similar triangles: Z = f * H_real / h_px
                Z_cam = (fx * H_real) / float(h_px_measured)
                print(f"    Measured pixel height: {h_px_measured:.1f}px -> estimated Z_cam={Z_cam:.3f} (splat units)")
            elif cam_vec is not None:
                Z_cam = cam_vec[2]
                print(f"    Using projected camera Z: {Z_cam:.3f}")
            else:
                print(f"    [WARN] No depth estimate available; falling back to heuristic")
                pixel_to_meter = (distance_to_window / 1.0) * (0.05 / 100.0)
                ctrl_y = error_x_px * pixel_to_meter
                ctrl_z = error_y_px * pixel_to_meter
                Z_cam = None

            if Z_cam is not None:
                # delta in camera frame (splat units)
                delta_x_cam = (error_x_px * Z_cam) / fx
                delta_y_cam = (error_y_px * Z_cam) / fy

                # camera frame: X right, Y down, Z forward
                delta_cam = np.array([delta_x_cam, delta_y_cam, 0.0])

                # transform camera delta to body frame, then to NED
                delta_body = self.pnp_estimator.R_cam_to_body @ delta_cam
                R_drone_ned = self.pnp_estimator._euler_to_rotation_matrix(
                    current_pose['rpy'][0], current_pose['rpy'][1], current_pose['rpy'][2]
                )
                delta_ned = R_drone_ned @ delta_body

                # Map corrections in NED: lateral = Y, vertical = Z
                ctrl_y = delta_ned[1]
                ctrl_z = delta_ned[2]

                print(f"    Reprojection-derived control: Y={ctrl_y:+.4f}m, Z={ctrl_z:+.4f}m (Z_cam used={Z_cam:.3f})")
            
            # Limit maximum step size
            max_step = 0.08  # Increased from 0.05m for faster convergence
            ctrl_magnitude = np.sqrt(ctrl_y**2 + ctrl_z**2)
            if ctrl_magnitude > max_step:
                scale = max_step / ctrl_magnitude
                ctrl_y *= scale
                ctrl_z *= scale
                print(f"    Scaled control (mag {ctrl_magnitude:.4f}m > {max_step}m): Y={ctrl_y:+.4f}m, Z={ctrl_z:+.4f}m")
            
            # Apply correction in NED frame
            target_pos = current_pose['position'].copy()
            target_pos[1] += ctrl_y  # Move right/left
            target_pos[2] += ctrl_z  # Move up/down
            
            # Bounds check
            target_pos_clipped = self.clip_position_to_bounds(target_pos)
            if not np.allclose(target_pos, target_pos_clipped):
                print(f"    [WARN] Target clipped to bounds")
                target_pos = target_pos_clipped
            
            # For small servo corrections, apply instantaneous open-loop pose update
            # This avoids dynamics transients and false-positive collisions for micro adjustments
            scaled_magnitude = np.sqrt(ctrl_y**2 + ctrl_z**2)
            if scaled_magnitude <= max_step:
                target_pos = current_pose['position'].copy()
                target_pos[1] += ctrl_y
                target_pos[2] += ctrl_z

                # Bounds check
                target_pos = self.clip_position_to_bounds(target_pos)

                # Log and apply instant update
                print(f"    [INFO] Applying instantaneous servo update to {target_pos}")
                current_pose['position'] = target_pos
                # Record frame at new pose without running dynamics
                rgb_tmp, _, _ = self.renderer.render(current_pose['position'], current_pose['rpy'])
                self.record_frame(rgb_tmp, pose=current_pose, annotation=f"SERVO_INSTANT_I{servo_iter}")
                print(f"    [OK] Instant move applied to {current_pose['position']}")
            else:
                # Collision check for larger moves and use dynamics-based motion
                if doesItCollide(target_pos):
                    print(f"    [FAIL] Collision detected!")
                    break

                print(f"    [INFO] Moving to {target_pos} (dynamics)")
                result = goToWaypoint(
                    current_pose, target_pos,
                    velocity=0.04,  # Slower for precision
                    pose_history=pose_history,
                    action=f'SERVO_W{self.window_count}_I{servo_iter}',
                    maintain_orientation=True,
                    navigator=self
                )

                if result == -1:
                    print(f"    [FAIL] Movement failed!")
                    break

                current_pose = result
                print(f"    [OK] Moved to {current_pose['position']}")
            
            # Visualize with projected window center
            rgb_new, _, _ = self.renderer.render(current_pose['position'], current_pose['rpy'])
            rgb_debug = rgb_new.copy()
            
            # Draw PnP-projected window center and detected centroid (if available)
            new_proj = self.pnp_estimator.project_window_to_pixel(window_3d_pos, current_pose)

            # Draw image center (red)
            cv2.drawMarker(rgb_debug, (int(img_center_x), int(img_center_y)),
                          (0, 0, 255), cv2.MARKER_CROSS, 50, 3)

            # Draw PnP projection (magenta)
            if new_proj is not None:
                px_pnp = (int(round(new_proj[0])), int(round(new_proj[1])))
                cv2.circle(rgb_debug, px_pnp, 14, (255, 0, 255), 3)
                cv2.putText(rgb_debug, 'PnP', (px_pnp[0]+10, px_pnp[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)

            # If the per-iteration detector returned a centroid, draw it (green)
            try:
                if 'center_iter' in locals() and center_iter is not None:
                    # center_iter may be (x,y) at original resolution
                    px_det = (int(round(center_iter[0])), int(round(center_iter[1])))
                    cv2.circle(rgb_debug, px_det, 20, (0, 255, 0), -1)
                    cv2.putText(rgb_debug, 'DET', (px_det[0]+10, px_det[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
                    # If PnP and DET disagree significantly, draw a line between them
                    if new_proj is not None:
                        dist_px = np.linalg.norm(np.array(px_pnp) - np.array(px_det))
                        if dist_px > 50:
                            cv2.line(rgb_debug, px_pnp, px_det, (0,255,255), 3)
                            cv2.putText(rgb_debug, f'Diff:{int(dist_px)}px', (px_det[0]+10, px_det[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            except Exception:
                pass

            # Draw error vector from image center to PnP (if available)
            if new_proj is not None:
                cv2.arrowedLine(rgb_debug, (int(img_center_x), int(img_center_y)),
                              (int(new_proj[0]), int(new_proj[1])), (255, 255, 0), 3, tipLength=0.1)
            
            self.record_frame(rgb_debug, pose=current_pose,
                            annotation=f"SERVO iter={servo_iter}, err_px=({error_x_px:.0f},{error_y_px:.0f})")
        
        if not servo_converged:
            print(f"\n  [WARN] Window centering did NOT converge after {max_servo_iterations} iterations")
            print(f"    Consider increasing max_servo_iterations or relaxing lateral_pixel_threshold")
        else:
            print(f"\n  [OK] Window successfully centered!")
        
        print(f"\n  [OK] Visual servoing complete")
        
        # ================================================================
        # PHASE 3: APPROACH AND PASS THROUGH
        # ================================================================
        print("\n[PHASE 3] Approach and Pass Through")
        
        # Calculate distance to window
        distance_to_window = np.linalg.norm(window_3d_pos - current_pose['position'])
        print(f"  Distance to window: {distance_to_window:.3f}m")
        
        approach_distance = 0.3  # Stop this far from window
        
        if distance_to_window > approach_distance:
            # Move forward toward window
            direction = window_3d_pos - current_pose['position']
            direction_norm = direction / np.linalg.norm(direction)
            approach_point = window_3d_pos - direction_norm * approach_distance
            
            # Bounds check
            if not self.is_position_in_bounds(approach_point):
                print(f"  [FAIL] Approach point outside bounds: {approach_point}")
                approach_point = self.clip_position_to_bounds(approach_point)
                print(f"  [INFO] Clipped to: {approach_point}")
            
            # Collision check
            if doesItCollide(approach_point):
                print(f"  [FAIL] Approach point collides!")
                return -1
            
            # Move to approach point
            result = goToWaypoint(
                current_pose, approach_point,
                velocity=0.05,
                pose_history=pose_history,
                action=f'APPROACH_W{self.window_count}',
                lock_roll_pitch=True,
                navigator=self
            )
            
            if result == -1:
                return -1
            
            current_pose = result
            print(f"  [OK] Approached to {approach_distance}m")
        
        # Pass through window
        through_distance = 1.0  # Distance past window
        direction = window_3d_pos - current_pose['position']
        direction_norm = direction / np.linalg.norm(direction)
        through_point = window_3d_pos + direction_norm * through_distance
        
        # Bounds check
        if not self.is_position_in_bounds(through_point):
            print(f"  [FAIL] Through point outside bounds: {through_point}")
            through_point = self.clip_position_to_bounds(through_point)
            print(f"  [INFO] Clipped to: {through_point}")
        
        # Collision check
        if doesItCollide(through_point):
            print(f"  [FAIL] Through point collides!")
            return -1
        
        # Pass through
        result = goToWaypoint(
            current_pose, through_point,
            velocity=0.08,
            pose_history=pose_history,
            action=f'PASS_THROUGH_W{self.window_count}',
            lock_roll_pitch=True,
            navigator=self
        )
        
        if result == -1:
            return -1
        
        current_pose = result
        print(f"  [OK] Passed through window!")
        
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
            print(f"  Approached")
        
        # Verification
        print(f"\n  Verification scan...")
        verified_window, _, verified_center, _ = self.scan_for_window(
            current_pose, pose_history, scan_type='verification'
        )
        
        if verified_window is None:
            print(f"  [ERROR] Verification failed")
            return -1
        
        print(f"  Verified")
        
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
        
        # STANDARD SINGLE-PANEL
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
        
        window_3d_pos, scan_frames, window_center_2d, corners_2d = navigator.scan_for_window(
            currentPose, pose_history, scan_type='initial'
        )
        
        if window_3d_pos is None:
            print(f"  [ERROR] Detection/PnP failed")
            break
        
        print(f"  Window at: {window_3d_pos}")
        
        print(f"  [OK] Window detected at: {window_3d_pos}")
        
        # Get window orientation from last PnP estimate
        window_rpy_ned = np.array([0.0, 0.0, 0.0])  # Placeholder - assume perpendicular
        
        # STEP 2: Visual servoing (orient, center, approach) - pass corners for visualization
        result = navigator.visual_servo_to_window(
            currentPose, window_3d_pos, window_rpy_ned, pose_history, initial_corners_2d=corners_2d, scan_frames=scan_frames
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