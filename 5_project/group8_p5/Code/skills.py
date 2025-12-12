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
from collisionChecker import doesItCollide


class NavigationSkills:
    """Container for all navigation skills"""
    
    def __init__(self, navigator):
        """
        Args:
            navigator: WindowNavigator instance with renderer, detector, pnp_estimator, etc.
        """
        self.nav = navigator
        self.yaw_error_initial = 0.0  # Store initial yaw error from FIX_YAW for VERIFY guidance
        
    # =========================================================================
    # SKILL 1: SCAN
    # =========================================================================
    def scan(self, current_pose, scan_type='initial', yaw_hint=None):
        """
        SKILL 1: SCAN
        Initial window detection using optical flow and PnP
        
        Args:
            current_pose: Current drone pose dict
            scan_type: Label for this scan ('initial', 'verify', etc.)
            yaw_hint: Initial yaw error (radians) for VERIFY directional selection
            
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
            current_pose, self.nav.pose_history, scan_type=scan_type, yaw_hint=yaw_hint
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
        
        # Store the detected area for later verification consistency checking
        if corners_2d is not None:
            # Calculate area from scan frames
            scan_frames = scan_data.get('frames', [])
            if len(scan_frames) > 0:
                mask, _, _, _ = self.nav.detector.detect_window(scan_frames)
                if mask is not None:
                    detected_area = np.sum(mask > 0.5)
                    self.nav.last_detected_area = detected_area
                    print(f"  Stored detection area: {detected_area:.0f} pixels for consistency checking")
        
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
        from navigation import wrap_angle  # Import here to avoid circular dependency
        
        print(f"\n{'='*60}")
        print(f"[SKILL: FIX_YAW]")
        print(f"{'='*60}")
        
        # Calculate desired yaw
        vec_to_window = window_3d_pos - current_pose['position']
        desired_yaw = np.arctan2(vec_to_window[1], vec_to_window[0])
        current_yaw = current_pose['rpy'][2]
        
        # yaw_error_initial already set in main.py before calling this function
        
        yaw_tolerance = np.radians(1.0)  # 1 degree
        yaw_step = np.radians(1.0)  # 1 degree per iteration
        max_yaw_iterations = 40
        
        yaw_iter = 0
        
        while yaw_iter < max_yaw_iterations:
            # Current yaw error
            yaw_error = wrap_angle(desired_yaw - current_pose['rpy'][2])
            
            if abs(yaw_error) < yaw_tolerance:
                print(f"  [OK] Yaw aligned after {yaw_iter} iterations")
                break
            
            # Yaw correction only
            step_yaw = np.clip(yaw_error, -yaw_step, yaw_step)
            current_pose['rpy'][2] += step_yaw
            
            # Log progress every 6 iterations
            if yaw_iter % 6 == 0:
                print(f"  Yaw iter {yaw_iter}: Yaw={np.degrees(current_pose['rpy'][2]):.1f}°, "
                      f"Error={np.degrees(yaw_error):.1f}°")
            
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
        Re-scan to verify alignment and confirm we're tracking the SAME window
        Uses area consistency to ensure correct window tracking
        
        Args:
            current_pose: Current drone pose dict
            window_3d_pos: Expected window position (from initial scan)
            cycle_num: Which align-verify cycle this is
            
        Returns:
            verified_window_pos: Updated window position, or None if failed
            corners_2d: Updated corners for depth estimation
            alignment_good: Bool, True if aligned well enough
            error_mag: Pixel error magnitude
        """
        print(f"\n  [SKILL: VERIFY] Cycle {cycle_num} - Re-scanning...")
        
        # Get the PREVIOUS area from last successful detection
        # This is stored in the navigator after initial scan
        previous_area = getattr(self.nav, 'last_detected_area', None)
        
        # Project the expected window position to get target pixel
        # This helps the detector prefer the CORRECT window if multiple are visible
        target_pixel = None
        proj_pixel = self.nav.pnp_estimator.project_window_to_pixel(window_3d_pos, current_pose)
        if proj_pixel is not None:
            target_pixel = (int(proj_pixel[0]), int(proj_pixel[1]))
            print(f"    Target pixel (projected from expected position): {target_pixel}")
        
        # Re-scan to get fresh detection
        # Pass target_pixel to prefer window closest to this location
        # Pass yaw_hint for directional selection: +ve = RIGHT, -ve = LEFT
        verified_window, corners_2d, scan_data = self.scan(
            current_pose, scan_type=f'verify_c{cycle_num}', yaw_hint=self.yaw_error_initial
        )
        
        # Get the area from this detection
        current_area = None
        if verified_window is not None and 'frames' in scan_data:
            # Re-detect to get the mask and calculate area
            scan_frames = scan_data['frames']
            mask, _, _, _ = self.nav.detector.detect_window(scan_frames, prefer_larger=True)
            if mask is not None:
                current_area = np.sum(mask > 0.5)
                print(f"    Current detection area: {current_area:.0f} pixels")
        
        # AREA CONSISTENCY CHECK
        if previous_area is not None and current_area is not None:
            area_ratio = current_area / previous_area
            area_change_pct = abs(area_ratio - 1.0) * 100
            
            print(f"    Previous area: {previous_area:.0f} pixels")
            print(f"    Area ratio: {area_ratio:.2f} (change: {area_change_pct:.1f}%)")
            
            # During yaw alignment, area should be relatively stable
            # Allow up to 100% change (2x or 0.5x) - beyond that it's likely wrong window
            if area_ratio > 2.0 or area_ratio < 0.5:
                print(f"    [ERROR] Area changed too much ({area_ratio:.2f}x)!")
                print(f"    This is likely a DIFFERENT WINDOW - rejecting!")
                
                # Try to re-scan with better guidance
                print(f"    [RETRY] Re-scanning with stricter target pixel guidance...")
                if target_pixel is not None:
                    verified_window, _, center_2d, corners_2d = self.nav.scan_for_window(
                        current_pose, self.nav.pose_history, 
                        scan_type=f'verify_c{cycle_num}_retry',
                        target_pixel=target_pixel,
                        yaw_hint=None  # Disable yaw hint, use target pixel only
                    )
                    
                    # Recalculate area
                    scan_waypoints_retry = self.nav.scanner.generate_scan_trajectory(current_pose)
                    scan_frames_retry = []
                    for wp in scan_waypoints_retry:
                        rgb_wp, _, _ = self.nav.renderer.render(wp['position'], wp['rpy'])
                        scan_frames_retry.append(rgb_wp)
                    
                    mask_retry, _, _, _ = self.nav.detector.detect_window(scan_frames_retry, prefer_larger=True)
                    if mask_retry is not None:
                        current_area = np.sum(mask_retry > 0.5)
                        area_ratio = current_area / previous_area
                        print(f"    Retry area ratio: {area_ratio:.2f}")
                        
                        if area_ratio > 2.0 or area_ratio < 0.5:
                            print(f"    [FAIL] Still wrong window after retry")
                            return None, None, False, float('inf')
                else:
                    return None, None, False, float('inf')
            else:
                print(f"    [OK] Area change within acceptable range")
        
        # Store current area for next verification
        if current_area is not None:
            self.nav.last_detected_area = current_area
        
        if verified_window is None:
            print(f"    [FAIL] Cannot see window")
            return None, None, False, float('inf')
        
        # CRITICAL: Check position consistency to prevent tracking DIFFERENT window
        pos_drift = np.linalg.norm(verified_window - window_3d_pos)
        print(f"    Position drift: {pos_drift:.3f} splat units")
        
        # Stricter on refinement cycles - must be same window!
        max_drift = 0.8 if cycle_num == 0 else 0.3
        
        if pos_drift > max_drift:
            print(f"    [ERROR] Position drift {pos_drift:.3f} > {max_drift:.3f}!")
            print(f"    Expected: {window_3d_pos}")
            print(f"    Detected: {verified_window}")
            print(f"    This is likely a DIFFERENT WINDOW - rejecting!")
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
        cv2.putText(rgb_annotated, f'C{cycle_num}', (px_proj[0]+25, px_proj[1]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        cv2.arrowedLine(rgb_annotated, (int(img_center_x), int(img_center_y)),
                      px_proj, (255, 255, 0), 3, tipLength=0.1)
        
        # Draw corners
        if corners_2d is not None:
            corners_int = corners_2d.astype(np.int32)
            cv2.polylines(rgb_annotated, [corners_int], True, (255, 255, 0), 2)
        
        self.nav.record_frame(rgb_annotated, pose=current_pose,
                            annotation=f"VERIFY_C{cycle_num}_ERR={error_mag:.0f}px")
        
        # Determine if aligned
        align_threshold = 50  # pixels
        alignment_good = error_mag < align_threshold
        
        if alignment_good:
            print(f"    [OK] Aligned (error < {align_threshold}px)")
        else:
            print(f"    [CONTINUE] Need more alignment")
        
        return verified_window, corners_2d, alignment_good, error_mag
    
    # =========================================================================
    # SKILL 5: APPROACH
    # =========================================================================
    def approach(self, current_pose, window_3d_pos):
        """
        SKILL 5: APPROACH
        Geometrically step forward through the aligned window
        Pure position updates, no controller dynamics
        
        Args:
            current_pose: Current drone pose dict (must be well-aligned!)
            window_3d_pos: Window position
            
        Returns:
            current_pose: Updated pose dict, or -1 if failed
        """
        print(f"\n{'='*60}")
        print(f"[SKILL: APPROACH] - Geometric Forward Stepping")
        print(f"{'='*60}")
        
        # Log alignment for debugging
        proj_pixel = self.nav.pnp_estimator.project_window_to_pixel(window_3d_pos, current_pose)
        
        if proj_pixel is not None:
            rgb, _, _ = self.nav.renderer.render(current_pose['position'], current_pose['rpy'])
            img_h, img_w = rgb.shape[:2]
            error_x = proj_pixel[0] - img_w / 2
            error_y = proj_pixel[1] - img_h / 2
            error_mag = np.sqrt(error_x**2 + error_y**2)
            
            print(f"  Pre-approach pixel error: {error_mag:.1f}px")
        
        # Calculate distance and direction to window
        distance = np.linalg.norm(window_3d_pos - current_pose['position'])
        print(f"  Distance to window: {distance:.3f} splat units")
        
        # Get FORWARD direction in body frame, then transform to NED
        current_yaw = current_pose['rpy'][2]
        
        # Forward direction in NED (body X-axis points forward)
        # In NED: X=North, Y=East, so forward direction depends on yaw
        forward_ned = np.array([
            np.cos(current_yaw),  # North component
            np.sin(current_yaw),  # East component
            0.0                   # No vertical component
        ])
        
        print(f"  Current yaw: {np.degrees(current_yaw):.1f} deg")
        print(f"  Forward direction (NED): [{forward_ned[0]:+.3f}, {forward_ned[1]:+.3f}, {forward_ned[2]:+.3f}]")
        
        # Move forward through window in small INCREMENTAL steps
        step_size = 0.03  # 0.03 splat units per step
        total_distance = distance + 0.15  # Go just 0.15 splat units past window (not 0.3!)
        num_steps = int(np.ceil(total_distance / step_size))
        num_steps = max(10, min(num_steps, 20))  # 10-20 steps
        
        # Use smaller drone radius for collision checking during approach
        # The drone is well-aligned, so we can be less conservative
        approach_radius = 0.02  # 2cm radius in splat units
        
        print(f"  Moving {total_distance:.3f} splat units forward in {num_steps} steps")
        print(f"  Step size: {step_size:.3f} splat units")
        print(f"  Using reduced collision radius: {approach_radius:.3f} splat units")
        
        # Track detection quality to detect window crossing
        initial_scan_frames = []
        initial_mask = None
        initial_area = 0
        peak_area = 0  # Track the maximum area we've seen
        
        # Get initial detection baseline (before starting approach)
        print(f"\n  Getting initial window detection baseline...")
        scan_waypoints_init = self.nav.scanner.generate_scan_trajectory(current_pose)
        for wp in scan_waypoints_init:
            rgb_init, _, _ = self.nav.renderer.render(wp['position'], wp['rpy'])
            initial_scan_frames.append(rgb_init)
        
        initial_mask, _, _, _ = self.nav.detector.detect_window(initial_scan_frames)
        if initial_mask is not None:
            initial_area = np.sum(initial_mask > 0.5)
            peak_area = initial_area  # Initialize peak
            print(f"  Initial window area: {initial_area:.0f} pixels")
        else:
            initial_area = 0
            print(f"  [WARN] Could not detect window initially")
        
        # Remember starting position
        start_pos = current_pose['position'].copy()
        
        for step in range(num_steps):
            # Calculate INCREMENTAL step (not cumulative!)
            step_increment = forward_ned * step_size
            new_pos = current_pose['position'] + step_increment
            
            # Clip to bounds
            new_pos = self.nav.clip_position_to_bounds(new_pos)
            
            # Update position (keep same orientation)
            current_pose['position'] = new_pos.copy()
            
            # Collision check with SMALLER radius
            if doesItCollide(new_pos, drone_radius=approach_radius):
                print(f"  [WARN] Step {step+1}/{num_steps} detected collision at {new_pos}")
                distance_to_window_now = np.linalg.norm(window_3d_pos - new_pos)
                print(f"  Distance to window: {distance_to_window_now:.3f}")
                # If we're reasonably close to the window, consider it success
                if distance_to_window_now < 0.8:  # More lenient threshold
                    print(f"  [OK] Close enough to window, considering approach successful")
                    break
                else:
                    print(f"  [ERROR] Too far from window, aborting")
                    return -1
            
            # Render and record frame
            rgb, _, _ = self.nav.renderer.render(current_pose['position'], current_pose['rpy'])
            self.nav.record_frame(rgb, pose=current_pose,
                                annotation=f"APPROACH_STEP_{step+1}/{num_steps}")
            
            print(f"  Step {step+1}/{num_steps}: pos={new_pos}")
            
            # VERIFICATION CHECK every 4 steps (starting from step 4)
            if (step + 1) % 4 == 0 and step >= 3:
                print(f"\n  [VERIFY] Checking window detection at step {step+1}...")
                
                # Quick scan from current position
                scan_waypoints_check = self.nav.scanner.generate_scan_trajectory(current_pose)
                scan_frames_check = []
                for wp in scan_waypoints_check:
                    rgb_check, _, _ = self.nav.renderer.render(wp['position'], wp['rpy'])
                    scan_frames_check.append(rgb_check)
                
                # Detect window
                check_mask, _, _, _ = self.nav.detector.detect_window(scan_frames_check)
                
                if check_mask is not None:
                    check_area = np.sum(check_mask > 0.5)
                    print(f"    Current window area: {check_area:.0f} pixels")
                    
                    # Update peak area
                    if check_area > peak_area:
                        peak_area = check_area
                        print(f"    New peak area: {peak_area:.0f} pixels")
                    
                    if peak_area > 0:
                        # Calculate ratio from PEAK (not initial)
                        # This detects when area starts DROPPING after reaching maximum
                        area_ratio_from_peak = check_area / peak_area
                        
                        print(f"    Area ratio from peak: {area_ratio_from_peak:.2f}")
                        print(f"    Peak area was: {peak_area:.0f} pixels")
                        
                        # CROSSING DETECTION: Area dropping from peak
                        # As we approach: area increases (window gets bigger)
                        # At closest point: area reaches PEAK
                        # As we pass through: area DROPS from peak
                        
                        if area_ratio_from_peak < 0.5 and step >= 8:
                            # Area dropped by >50% from peak after at least 8 steps → We've passed through!
                            print(f"    [OK] Window area dropped {(1-area_ratio_from_peak)*100:.0f}% from peak - passed through!")
                            print(f"    Stopping at step {step+1}/{num_steps}")
                            break
                        elif area_ratio_from_peak < 0.6 and step >= 12:
                            # Area dropped by >40% from peak after many steps → likely passed
                            print(f"    [OK] Window area dropped {(1-area_ratio_from_peak)*100:.0f}% from peak after {step+1} steps - likely passed!")
                            print(f"    Stopping at step {step+1}/{num_steps}")
                            break
                        else:
                            # Still approaching or at peak
                            print(f"    [CONTINUE] Window area at {area_ratio_from_peak*100:.0f}% of peak, continuing")
                else:
                    print(f"    [WARN] Cannot detect window - may have passed through")
                    # If we can't detect window anymore after several steps, we probably passed it
                    if step >= 8:
                        print(f"    [OK] Lost window tracking after {step+1} steps - assuming passed through")
                        break
        
        print(f"  [OK] Approach complete - moved forward {total_distance:.3f}m")
        print(f"  Final position (NED): {current_pose['position']}")
        
        return current_pose
    
    # =========================================================================
    # SKILL 6: RECENTER
    # =========================================================================
    def recenter(self, current_pose):
        """
        SKILL 6: RECENTER
        Iteratively reset Y, Z, Yaw, Pitch, Roll to zero while maintaining LOCAL X
        LOCAL X = actual forward distance traveled through window (correct!)
        Resetting Y/Yaw just re-aligns us to centerline
        Saves frames during the process
        
        Args:
            current_pose: Current drone pose dict
            
        Returns:
            current_pose: Updated pose dict
        """
        from navigation import wrap_angle
        
        print(f"\n{'='*60}")
        print(f"[SKILL: RECENTER]")
        print(f"{'='*60}")
        
        # Keep the LOCAL X we actually traveled - this is correct!
        # During approach, we moved forward through the window
        # Now we just need to re-align to centerline (Y=0, yaw=0)
        target_x = current_pose['position'][0]  # LOCAL forward distance traveled
        
        current_global_x = current_pose['position'][0]
        current_global_y = current_pose['position'][1]
        current_global_z = current_pose['position'][2]
        
        print(f"  Goal: Keep X={target_x:.3f} (forward distance traveled), reset Y/Z/RPY to zero")
        print(f"  Starting pose:")
        print(f"    Position: [{current_global_x:.3f}, {current_global_y:.3f}, {current_global_z:.3f}]")
        print(f"    RPY (deg): [{np.degrees(current_pose['rpy'][0]):.1f}, {np.degrees(current_pose['rpy'][1]):.1f}, {np.degrees(current_pose['rpy'][2]):.1f}]")
        
        # SAFETY: Move 0.1 splat units forward first to clear window 1
        print(f"\n  [SAFETY] Moving 0.1 splat units forward to clear window 1...")
        safety_distance = 0.1
        safety_steps = 10  # 0.01 per step
        step_size = safety_distance / safety_steps
        
        for safety_step in range(safety_steps):
            # Move forward in current yaw direction
            current_yaw = current_pose['rpy'][2]
            forward_increment = np.array([
                np.cos(current_yaw) * step_size,  # X (North)
                np.sin(current_yaw) * step_size,  # Y (East)
                0.0                                # Z (no vertical)
            ])
            
            current_pose['position'] += forward_increment
            current_pose['position'] = self.nav.clip_position_to_bounds(current_pose['position'])
            
            # Check collision during safety movement
            if doesItCollide(current_pose['position']):
                print(f"    [WARN] Safety step {safety_step+1}: Collision detected at {current_pose['position']}")
            
            # Render frame
            rgb, _, _ = self.nav.renderer.render(current_pose['position'], current_pose['rpy'])
            self.nav.record_frame(rgb, pose=current_pose,
                                annotation=f"RECENTER_SAFETY_{safety_step+1}/{safety_steps}")
        
        print(f"  Safety movement complete. New position: {current_pose['position']}")
        print(f"  Total forward distance: {current_pose['position'][0]:.3f} splat units")
        
        # Update target_x to new position after safety movement
        target_x = current_pose['position'][0]
        print(f"\n  Now starting lateral/yaw recenter with X={target_x:.3f}")
        
        # Target state: keep X (after safety move), zero everything else
        target_y = 0.0
        target_z = 0.0
        target_roll = 0.0
        target_pitch = 0.0
        target_yaw = 0.0
        
        # Iterative convergence parameters
        max_iterations = 50
        position_tolerance = 0.01  # 1cm (relaxed from 5mm)
        angle_tolerance = np.radians(2.0)  # 2 degrees (relaxed from 1)
        
        # Step sizes for gradual convergence (smaller initial steps!)
        position_step = 0.01  # 1cm per step (reduced from 2cm)
        angle_step = np.radians(1.0)  # 1 degree per step (reduced from 2)
        
        # Check if already well-centered
        initial_y_error = abs(current_pose['position'][1] - target_y)
        initial_z_error = abs(current_pose['position'][2] - target_z)
        initial_yaw_error = abs(wrap_angle(target_yaw - current_pose['rpy'][2]))
        
        if (initial_y_error < position_tolerance and 
            initial_z_error < position_tolerance and
            initial_yaw_error < angle_tolerance):
            print(f"  [OK] Already centered, skipping recenter")
            return current_pose
        
        for iteration in range(max_iterations):
            # Calculate errors
            error_y = target_y - current_pose['position'][1]
            error_z = target_z - current_pose['position'][2]
            error_roll = wrap_angle(target_roll - current_pose['rpy'][0])
            error_pitch = wrap_angle(target_pitch - current_pose['rpy'][1])
            error_yaw = wrap_angle(target_yaw - current_pose['rpy'][2])
            
            # Check convergence
            pos_converged = abs(error_y) < position_tolerance and abs(error_z) < position_tolerance
            ang_converged = (abs(error_roll) < angle_tolerance and 
                           abs(error_pitch) < angle_tolerance and 
                           abs(error_yaw) < angle_tolerance)
            
            if pos_converged and ang_converged:
                print(f"  [OK] Recentered after {iteration} iterations")
                break
            
            # Apply corrections with step limits
            # Position corrections
            if abs(error_y) > position_tolerance:
                step_y = np.clip(error_y, -position_step, position_step)
                current_pose['position'][1] += step_y
            
            if abs(error_z) > position_tolerance:
                step_z = np.clip(error_z, -position_step, position_step)
                current_pose['position'][2] += step_z
            
            # Angle corrections
            if abs(error_roll) > angle_tolerance:
                step_roll = np.clip(error_roll, -angle_step, angle_step)
                current_pose['rpy'][0] += step_roll
            
            if abs(error_pitch) > angle_tolerance:
                step_pitch = np.clip(error_pitch, -angle_step, angle_step)
                current_pose['rpy'][1] += step_pitch
            
            if abs(error_yaw) > angle_tolerance:
                step_yaw = np.clip(error_yaw, -angle_step, angle_step)
                current_pose['rpy'][2] += step_yaw
            
            # Clip position to bounds (keep X fixed)
            current_pose['position'] = self.nav.clip_position_to_bounds(current_pose['position'])
            current_pose['position'][0] = target_x  # Force X to remain constant
            
            # Collision check - ENABLED to observe if we hit window 1 during recenter
            is_colliding = doesItCollide(current_pose['position'])
            
            if is_colliding:
                print(f"  [COLLISION] Iteration {iteration}: Detected at position {current_pose['position']}")
                print(f"    Y={current_pose['position'][1]:+.3f}, Z={current_pose['position'][2]:+.3f}")
                print(f"    This means we're hitting window 1 while moving back to centerline!")
                # Don't break - continue to see the full collision pattern
            
            # Render and save frame every iteration
            rgb, _, _ = self.nav.renderer.render(current_pose['position'], current_pose['rpy'])
            self.nav.record_frame(rgb, pose=current_pose,
                                annotation=f"RECENTER_iter{iteration}")
            
            # Log progress every iteration to track collision pattern
            collision_status = "COLLISION" if is_colliding else "clear"
            print(f"  Iter {iteration}: Y={current_pose['position'][1]:+.3f}, Z={current_pose['position'][2]:+.3f}, "
                  f"Yaw={np.degrees(current_pose['rpy'][2]):+.1f}° [{collision_status}]")
        
        print(f"  Final pose:")
        print(f"    Position: [{current_pose['position'][0]:.3f}, {current_pose['position'][1]:.3f}, {current_pose['position'][2]:.3f}]")
        print(f"    RPY (deg): [{np.degrees(current_pose['rpy'][0]):.1f}, {np.degrees(current_pose['rpy'][1]):.1f}, {np.degrees(current_pose['rpy'][2]):.1f}]")
        print(f"  [OK] Recenter complete")
        
        return current_pose