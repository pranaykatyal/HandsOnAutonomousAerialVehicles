"""
Drone Navigation Skills Module

6 core skills for window navigation:
1. SCAN - Initial window detection
2. FIX_YAW - Coarse yaw (and Z) alignment
3. ALIGN - Fine Y/Z position correction
4. VERIFY - Re-detect window and check alignment
5. APPROACH - Move through aligned window
6. RECENTER - Reset pose for next window
"""

import numpy as np
import cv2
from navigation import wrap_angle
from collisionChecker import doesItCollide


class NavigationSkills:
    """Container for all navigation skills"""
    
    def __init__(self, navigator):
        """
        Args:
            navigator: WindowNavigator instance with renderer, detector, pnp_estimator, etc.
        """
        self.nav = navigator
        
    # =========================================================================
    # SKILL 1: SCAN
    # =========================================================================
    def scan(self, current_pose, scan_type='initial'):
        """
        SKILL 1: SCAN
        Initial window detection using optical flow and PnP
        
        Args:
            current_pose: Current drone pose dict
            scan_type: Label for this scan ('initial', 'verify', etc.)
            
        Returns:
            window_3d_pos: (3,) Window position in NED, or None if failed
            corners_2d: (4,2) PnP corners for depth estimation
            scan_data: Dict with frames and other info
        """
        print(f"\n{'='*60}")
        print(f"[SKILL: SCAN] Type: {scan_type}")
        print(f"{'='*60}")
        
        scan_label = f"{scan_type}_window{self.nav.window_count}_scan{self.nav.scan_count}"
        
        # Generate scan trajectory
        scan_waypoints = self.nav.scanner.generate_scan_trajectory(current_pose)
        print(f"  Generated {len(scan_waypoints)} scan waypoints")
        
        # Capture frames
        scan_frames = []
        for i, waypoint in enumerate(scan_waypoints):
            rgb, _, _ = self.nav.renderer.render(waypoint['position'], waypoint['rpy'])
            scan_frames.append(rgb)
            print(f"    Frame {i+1}/{len(scan_waypoints)}")
        
        # Detect window
        print(f"  Detecting window...")
        window_3d_pos, _, center_2d, corners_2d = self.nav.scan_for_window(
            current_pose, self.nav.pose_history, scan_type=scan_type
        )
        
        scan_data = {
            'frames': scan_frames,
            'center_2d': center_2d,
            'type': scan_type
        }
        
        if window_3d_pos is None:
            print(f"  [FAIL] Window not detected")
            return None, None, scan_data
        
        print(f"  [OK] Window detected at: {window_3d_pos}")
        return window_3d_pos, corners_2d, scan_data
    
    # =========================================================================
    # SKILL 2: FIX_YAW
    # =========================================================================
    def fix_yaw(self, current_pose, window_3d_pos):
        """
        SKILL 2: FIX_YAW
        Rotate in place to face window, also correct Z during rotation
        
        Args:
            current_pose: Current drone pose dict
            window_3d_pos: Target window position
            
        Returns:
            current_pose: Updated pose dict, or -1 if failed
        """
        print(f"\n{'='*60}")
        print(f"[SKILL: FIX_YAW]")
        print(f"{'='*60}")
        
        # Calculate desired yaw
        vec_to_window = window_3d_pos - current_pose['position']
        desired_yaw = np.arctan2(vec_to_window[1], vec_to_window[0])
        current_yaw = current_pose['rpy'][2]
        
        yaw_tolerance = np.radians(1.0)  # 1 degree
        yaw_step = np.radians(1.0)  # 1 degree per iteration
        max_yaw_iterations = 40
        
        z_threshold_px = 20  # Start correcting Z when vertical error > 20px
        z_step_max = 0.02  # Max Z correction per iteration
        
        yaw_iter = 0
        
        while yaw_iter < max_yaw_iterations:
            # Current yaw error
            yaw_error = wrap_angle(desired_yaw - current_pose['rpy'][2])
            
            if abs(yaw_error) < yaw_tolerance:
                print(f"  [OK] Yaw aligned after {yaw_iter} iterations")
                break
            
            # Yaw correction
            step_yaw = np.clip(yaw_error, -yaw_step, yaw_step)
            current_pose['rpy'][2] += step_yaw
            
            # Z correction during yaw rotation
            proj_pixel = self.nav.pnp_estimator.project_window_to_pixel(
                window_3d_pos, current_pose
            )
            
            if proj_pixel is not None:
                rgb, _, _ = self.nav.renderer.render(current_pose['position'], current_pose['rpy'])
                img_h, img_w = rgb.shape[:2]
                error_y_px = proj_pixel[1] - img_h / 2
                
                if abs(error_y_px) > z_threshold_px and yaw_iter > 5:
                    # Correct Z
                    Z_cam = 0.3  # Approximate
                    fy = self.nav.pnp_estimator.camera_matrix_original[1, 1]
                    delta_y_cam = (error_y_px * Z_cam) / fy
                    
                    # Transform to NED
                    delta_cam = np.array([0.0, delta_y_cam, 0.0])
                    delta_body = self.nav.pnp_estimator.R_cam_to_body @ delta_cam
                    R_drone_ned = self.nav.pnp_estimator._euler_to_rotation_matrix(
                        current_pose['rpy'][0], current_pose['rpy'][1], current_pose['rpy'][2]
                    )
                    delta_ned = R_drone_ned @ delta_body
                    z_correction = np.clip(delta_ned[2], -z_step_max, z_step_max)
                    
                    current_pose['position'][2] += z_correction
                    current_pose['position'] = self.nav.clip_position_to_bounds(current_pose['position'])
                    
                    print(f"  Yaw iter {yaw_iter}: Yaw={np.degrees(current_pose['rpy'][2]):.1f}°, "
                          f"Error={np.degrees(yaw_error):.1f}°, Z_corr={z_correction:+.3f}")
            
            # Record frame
            rgb, _, _ = self.nav.renderer.render(current_pose['position'], current_pose['rpy'])
            self.nav.record_frame(rgb, pose=current_pose, annotation=f"FIX_YAW_{yaw_iter}")
            
            yaw_iter += 1
        
        if yaw_iter >= max_yaw_iterations:
            print(f"  [FAIL] Yaw alignment exceeded max iterations")
            return -1
        
        return current_pose
    
    # =========================================================================
    # SKILL 3: ALIGN
    # =========================================================================
    def align(self, current_pose, window_3d_pos, corners_2d=None):
        """
        SKILL 3: ALIGN
        Fine Y/Z position correction using reprojection
        
        Args:
            current_pose: Current drone pose dict
            window_3d_pos: Target window position
            corners_2d: Optional corners for depth estimation
            
        Returns:
            current_pose: Updated pose dict
            error_mag: Final pixel error magnitude
        """
        print(f"\n  [SKILL: ALIGN]")
        
        # Project window
        proj_pixel = self.nav.pnp_estimator.project_window_to_pixel(window_3d_pos, current_pose)
        
        if proj_pixel is None:
            print(f"    [WARN] Cannot project window")
            return current_pose, float('inf')
        
        # Measure error
        rgb, _, _ = self.nav.renderer.render(current_pose['position'], current_pose['rpy'])
        img_h, img_w = rgb.shape[:2]
        img_center_x = img_w / 2
        img_center_y = img_h / 2
        
        error_x_px = proj_pixel[0] - img_center_x
        error_y_px = proj_pixel[1] - img_center_y
        error_mag = np.sqrt(error_x_px**2 + error_y_px**2)
        
        print(f"    Current error: ({error_x_px:+.1f}, {error_y_px:+.1f})px, mag={error_mag:.1f}px")
        
        # Estimate depth
        if corners_2d is not None:
            try:
                Z_cam = self.nav.estimate_depth_from_height(corners_2d)
            except:
                Z_cam = 0.3
        else:
            Z_cam = 0.3
        
        # Compute corrections
        fx = self.nav.pnp_estimator.camera_matrix_original[0, 0]
        fy = self.nav.pnp_estimator.camera_matrix_original[1, 1]
        
        delta_x_cam = (error_x_px * Z_cam) / fx
        delta_y_cam = (error_y_px * Z_cam) / fy
        delta_cam = np.array([delta_x_cam, delta_y_cam, 0.0])
        
        # Transform to NED
        delta_body = self.nav.pnp_estimator.R_cam_to_body @ delta_cam
        R_drone_ned = self.nav.pnp_estimator._euler_to_rotation_matrix(
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
        new_pos = self.nav.clip_position_to_bounds(new_pos)
        
        # Collision check
        if doesItCollide(new_pos):
            print(f"    [WARN] Correction would cause collision, skipping")
            return current_pose, error_mag
        
        current_pose['position'] = new_pos
        print(f"    [OK] Moved to {current_pose['position']}")
        
        return current_pose, error_mag
    
    # =========================================================================
    # SKILL 4: VERIFY
    # =========================================================================
    def verify(self, current_pose, window_3d_pos, cycle_num=0):
        """
        SKILL 4: VERIFY
        Re-detect window and check alignment quality
        
        Args:
            current_pose: Current drone pose dict
            window_3d_pos: Expected window position
            cycle_num: Which align-verify cycle this is
            
        Returns:
            verified_window_pos: Updated window position, or None if failed
            corners_2d: Updated corners for depth estimation
            alignment_good: Bool, True if aligned well enough
            error_mag: Pixel error magnitude
        """
        print(f"\n  [SKILL: VERIFY] Cycle {cycle_num}")
        
        # Re-detect window
        verified_window, corners_2d, scan_data = self.scan(
            current_pose, scan_type=f'verify_cycle{cycle_num}'
        )
        
        if verified_window is None:
            print(f"    [FAIL] Cannot see window")
            return None, None, False, float('inf')
        
        # Check position consistency to prevent false detections
        pos_drift = np.linalg.norm(verified_window - window_3d_pos)
        print(f"    Position drift: {pos_drift:.3f} splat units")
        
        # CRITICAL: Reject if window jumped too far (likely different window)
        # Allow larger drift on first cycle (initial detection), smaller on refinement
        max_drift = 0.8 if cycle_num == 0 else 0.3
        
        if pos_drift > max_drift:
            print(f"    [ERROR] Position drift {pos_drift:.3f} > {max_drift:.3f}!")
            print(f"    Original window: {window_3d_pos}")
            print(f"    Detected window: {verified_window}")
            print(f"    This is likely a DIFFERENT window - rejecting!")
            return None, None, False, float('inf')
        
        # Check alignment
        proj_pixel = self.nav.pnp_estimator.project_window_to_pixel(verified_window, current_pose)
        
        if proj_pixel is None:
            print(f"    [WARN] Cannot project window")
            return verified_window, corners_2d, False, float('inf')
        
        rgb, _, _ = self.nav.renderer.render(current_pose['position'], current_pose['rpy'])
        img_h, img_w = rgb.shape[:2]
        img_center_x = img_w / 2
        img_center_y = img_h / 2
        
        error_x_px = proj_pixel[0] - img_center_x
        error_y_px = proj_pixel[1] - img_center_y
        error_mag = np.sqrt(error_x_px**2 + error_y_px**2)
        
        print(f"    Alignment error: ({error_x_px:+.1f}, {error_y_px:+.1f})px, mag={error_mag:.1f}px")
        
        # Visualize
        rgb_annotated = rgb.copy()
        cv2.drawMarker(rgb_annotated, (int(img_center_x), int(img_center_y)),
                      (0, 0, 255), cv2.MARKER_CROSS, 50, 3)
        
        px_proj = (int(round(proj_pixel[0])), int(round(proj_pixel[1])))
        cv2.circle(rgb_annotated, px_proj, 20, (255, 0, 255), 3)
        cv2.putText(rgb_annotated, f'Cycle {cycle_num}', (px_proj[0]+25, px_proj[1]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        cv2.arrowedLine(rgb_annotated, (int(img_center_x), int(img_center_y)),
                      px_proj, (255, 255, 0), 3, tipLength=0.1)
        
        # Draw corners if available
        if corners_2d is not None:
            corners_int = corners_2d.astype(np.int32)
            cv2.polylines(rgb_annotated, [corners_int], True, (255, 255, 0), 2)
        
        self.nav.record_frame(rgb_annotated, pose=current_pose,
                            annotation=f"VERIFY_C{cycle_num}_ERR={error_mag:.0f}px")
        
        # Determine if alignment is good
        align_threshold = 50  # pixels
        alignment_good = error_mag < align_threshold
        
        if alignment_good:
            print(f"    [OK] Alignment verified (error < {align_threshold}px)")
        else:
            print(f"    [WARN] Alignment needs improvement (error ≥ {align_threshold}px)")
        
        return verified_window, corners_2d, alignment_good, error_mag
    
    # =========================================================================
    # SKILL 5: APPROACH
    # =========================================================================
    def approach(self, current_pose, window_3d_pos):
        """
        SKILL 5: APPROACH
        Move through the aligned window
        
        Args:
            current_pose: Current drone pose dict (must be well-aligned!)
            window_3d_pos: Window position
            
        Returns:
            current_pose: Updated pose dict, or -1 if failed
        """
        print(f"\n{'='*60}")
        print(f"[SKILL: APPROACH]")
        print(f"{'='*60}")
        
        # Final alignment check
        proj_pixel = self.nav.pnp_estimator.project_window_to_pixel(window_3d_pos, current_pose)
        
        if proj_pixel is not None:
            rgb, _, _ = self.nav.renderer.render(current_pose['position'], current_pose['rpy'])
            img_h, img_w = rgb.shape[:2]
            error_x = proj_pixel[0] - img_w / 2
            error_y = proj_pixel[1] - img_h / 2
            error_mag = np.sqrt(error_x**2 + error_y**2)
            
            print(f"  Pre-approach alignment: {error_mag:.1f}px")
            
            if error_mag > 100:
                print(f"  [ERROR] Alignment too poor to approach safely!")
                return -1
        
        # Calculate distance
        distance = np.linalg.norm(window_3d_pos - current_pose['position'])
        print(f"  Distance to window: {distance:.3f} splat units")
        
        # Approach to 0.2 units from window
        approach_distance = 0.2
        
        if distance > approach_distance:
            # Move forward along camera axis
            R_drone_ned = self.nav.pnp_estimator._euler_to_rotation_matrix(
                current_pose['rpy'][0], current_pose['rpy'][1], current_pose['rpy'][2]
            )
            forward_direction = R_drone_ned @ np.array([1, 0, 0])
            
            move_distance = distance - approach_distance
            approach_point = current_pose['position'] + forward_direction * move_distance
            
            print(f"  Moving forward {move_distance:.3f} units")
            
            # Bounds and collision check
            approach_point = self.nav.clip_position_to_bounds(approach_point)
            if doesItCollide(approach_point):
                print(f"  [ERROR] Approach point collides!")
                return -1
            
            # Move using goToWaypoint
            from navigation import goToWaypoint
            result = goToWaypoint(
                current_pose, approach_point,
                velocity=0.03,
                pose_history=self.nav.pose_history,
                action=f'APPROACH_W{self.nav.window_count}',
                lock_roll_pitch=True,
                navigator=self.nav
            )
            
            if result == -1:
                print(f"  [ERROR] Approach failed")
                return -1
            
            current_pose = result
            print(f"  [OK] Approached to {approach_distance} units")
        
        # Pass through
        print(f"  Passing through window...")
        through_distance = 0.5
        
        R_drone_ned = self.nav.pnp_estimator._euler_to_rotation_matrix(
            current_pose['rpy'][0], current_pose['rpy'][1], current_pose['rpy'][2]
        )
        forward_direction = R_drone_ned @ np.array([1, 0, 0])
        through_point = current_pose['position'] + forward_direction * through_distance
        
        through_point = self.nav.clip_position_to_bounds(through_point)
        if doesItCollide(through_point):
            print(f"  [ERROR] Pass-through point collides!")
            return -1
        
        from navigation import goToWaypoint
        result = goToWaypoint(
            current_pose, through_point,
            velocity=0.05,
            pose_history=self.nav.pose_history,
            action=f'PASS_THROUGH_W{self.nav.window_count}',
            lock_roll_pitch=True,
            navigator=self.nav
        )
        
        if result == -1:
            print(f"  [ERROR] Pass-through failed")
            return -1
        
        current_pose = result
        print(f"  [OK] Passed through window!")
        
        return current_pose
    
    # =========================================================================
    # SKILL 6: RECENTER
    # =========================================================================
    def recenter(self, current_pose):
        """
        SKILL 6: RECENTER
        Reset yaw, Y, Z to prepare for next window
        
        Args:
            current_pose: Current drone pose dict
            
        Returns:
            current_pose: Updated pose dict with reset orientation/position
        """
        print(f"\n{'='*60}")
        print(f"[SKILL: RECENTER]")
        print(f"{'='*60}")
        
        # Reset yaw to 0
        print(f"  Resetting yaw from {np.degrees(current_pose['rpy'][2]):.1f}° to 0°")
        from navigation import goToWaypoint_yaw
        result = goToWaypoint_yaw(
            current_pose, 0.0,
            pose_history=self.nav.pose_history,
            action=f'RECENTER_YAW_W{self.nav.window_count}',
            navigator=self.nav
        )
        
        if result != -1:
            current_pose = result
        
        # Reset Y and Z toward 0
        target_y = 0.0
        target_z = 0.0
        
        current_y = current_pose['position'][1]
        current_z = current_pose['position'][2]
        
        if abs(current_y) > 0.1 or abs(current_z) > 0.1:
            print(f"  Recentering position from Y={current_y:.2f}, Z={current_z:.2f} to Y=0, Z=0")
            
            target_pos = current_pose['position'].copy()
            target_pos[1] = target_y
            target_pos[2] = target_z
            
            target_pos = self.nav.clip_position_to_bounds(target_pos)
            
            if not doesItCollide(target_pos):
                from navigation import goToWaypoint
                result = goToWaypoint(
                    current_pose, target_pos,
                    velocity=0.05,
                    pose_history=self.nav.pose_history,
                    action=f'RECENTER_POS_W{self.nav.window_count}',
                    maintain_orientation=True,
                    navigator=self.nav
                )
                
                if result != -1:
                    current_pose = result
        
        print(f"  [OK] Recentered to Y={current_pose['position'][1]:.2f}, "
              f"Z={current_pose['position'][2]:.2f}, Yaw={np.degrees(current_pose['rpy'][2]):.1f}°")
        
        return current_pose