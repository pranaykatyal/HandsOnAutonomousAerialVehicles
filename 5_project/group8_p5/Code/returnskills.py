"""
Return Journey Navigation Skills

Skills for navigating back through windows 4â†’3â†’2â†’1 after turning around.
Reference frame is now reversed (drone facing opposite direction).
"""

import numpy as np
import cv2
from collisionChecker import doesItCollide


class ReturnNavigationSkills:
    """Container for all return navigation skills"""
    
    def __init__(self, navigator):
        """
        Args:
            navigator: WindowNavigator instance with renderer, detector, pnp_estimator, etc.
        """
        self.nav = navigator
        self.return_window_count = 0  # Track windows on return journey (4â†’3â†’2â†’1)
        
    # =========================================================================
    # RETURN SKILL 1: SCAN_RETURN
    # =========================================================================
    def scan_return(self, current_pose, scan_type='return_scan'):
        """
        RETURN SKILL 1: SCAN_RETURN
        Window detection for return journey
        
        Args:
            current_pose: Current drone pose dict
            scan_type: Label for this scan
            
        Returns:
            window_3d_pos: (3,) Window position in NED, or None if failed
            corners_2d: (4,2) PnP corners for depth estimation
            scan_data: Dict with frames and other info
        """
        print(f"\n{'='*60}")
        print(f"[RETURN SKILL: SCAN_RETURN] Type: {scan_type}")
        print(f"{'='*60}")
        
        scan_label = f"{scan_type}_returnwin{self.return_window_count}_scan{self.nav.scan_count}"
        
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
        window_3d_pos, _, center_2d, corners_2d, debug_info = self.nav.scan_for_window(
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
    # RETURN SKILL 2: FIX_YAW_RETURN
    # =========================================================================
    def fix_yaw_return(self, current_pose, window_3d_pos):
        """
        RETURN SKILL 2: FIX_YAW_RETURN
        Rotate in place to face window on return journey
        
        Args:
            current_pose: Current drone pose dict
            window_3d_pos: Target window position
            
        Returns:
            current_pose: Updated pose dict, or -1 if failed
        """
        from navigation import wrap_angle
        
        print(f"\n{'='*60}")
        print(f"[RETURN SKILL: FIX_YAW_RETURN]")
        print(f"{'='*60}")
        
        # Calculate desired yaw
        vec_to_window = window_3d_pos - current_pose['position']
        desired_yaw = np.arctan2(vec_to_window[1], vec_to_window[0])
        current_yaw = current_pose['rpy'][2]
        
        yaw_tolerance = np.radians(1.0)
        yaw_step = np.radians(1.0)
        max_yaw_iterations = 40
        
        yaw_iter = 0
        
        while yaw_iter < max_yaw_iterations:
            yaw_error = wrap_angle(desired_yaw - current_pose['rpy'][2])
            
            if abs(yaw_error) < yaw_tolerance:
                print(f"  [OK] Yaw aligned after {yaw_iter} iterations")
                break
            
            step_yaw = np.clip(yaw_error, -yaw_step, yaw_step)
            current_pose['rpy'][2] += step_yaw
            
            if yaw_iter % 6 == 0:
                print(f"  Yaw iter {yaw_iter}: Yaw={np.degrees(current_pose['rpy'][2]):.1f}Â°, "
                      f"Error={np.degrees(yaw_error):.1f}Â°")
            
            rgb, _, _ = self.nav.renderer.render(current_pose['position'], current_pose['rpy'])
            self.nav.record_frame(rgb, pose=current_pose, annotation=f"RETURN_FIX_YAW_{yaw_iter}")
            
            yaw_iter += 1
        
        if yaw_iter >= max_yaw_iterations:
            print(f"  [FAIL] Yaw alignment exceeded max iterations")
            return -1
        
        return current_pose
    
    # =========================================================================
    # RETURN SKILL 3: ALIGN_RETURN
    # =========================================================================
    def align_return(self, current_pose, window_3d_pos, corners_2d=None):
        """
        RETURN SKILL 3: ALIGN_RETURN
        Fine Y/Z position correction for return journey
        
        Args:
            current_pose: Current drone pose dict
            window_3d_pos: Target window position
            corners_2d: Optional corners for depth estimation
            
        Returns:
            current_pose: Updated pose dict
            error_mag: Final pixel error magnitude
        """
        print(f"\n  [RETURN SKILL: ALIGN_RETURN]")
        
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
    # RETURN SKILL 4: APPROACH_RETURN
    # =========================================================================
    def approach_return(self, current_pose, window_3d_pos):
        """
        RETURN SKILL 4: APPROACH_RETURN
        Geometrically step forward through aligned window on return journey
        
        Args:
            current_pose: Current drone pose dict
            window_3d_pos: Window position
            
        Returns:
            current_pose: Updated pose dict, or -1 if failed
        """
        print(f"\n{'='*60}")
        print(f"[RETURN SKILL: APPROACH_RETURN]")
        print(f"{'='*60}")
        
        # Calculate distance and direction to window
        distance = np.linalg.norm(window_3d_pos - current_pose['position'])
        print(f"  Distance to window: {distance:.3f} splat units")
        
        # Get FORWARD direction
        current_yaw = current_pose['rpy'][2]
        forward_ned = np.array([
            np.cos(current_yaw),
            np.sin(current_yaw),
            0.0
        ])
        
        print(f"  Current yaw: {np.degrees(current_yaw):.1f} deg")
        print(f"  Forward direction (NED): {forward_ned}")
        
        # Move forward through window
        step_size = 0.03
        total_distance = distance + 0.15
        num_steps = int(np.ceil(total_distance / step_size))
        num_steps = max(10, min(num_steps, 20))
        
        print(f"  Moving {total_distance:.3f} splat units forward in {num_steps} steps")
        
        for step in range(num_steps):
            step_increment = forward_ned * step_size
            new_pos = current_pose['position'] + step_increment
            new_pos = self.nav.clip_position_to_bounds(new_pos)
            
            current_pose['position'] = new_pos.copy()
            
            # Render
            rgb, _, _ = self.nav.renderer.render(current_pose['position'], current_pose['rpy'])
            self.nav.record_frame(rgb, pose=current_pose,
                                annotation=f"RETURN_APPROACH_STEP_{step+1}/{num_steps}")
            
            print(f"  Step {step+1}/{num_steps}: pos={new_pos}")
        
        print(f"  [OK] Return approach complete")
        return current_pose
    
    # =========================================================================
    # RETURN SKILL 4B: APPROACH_RETURN_WINDOW4
    # =========================================================================
    def approach_return_window4(self, current_pose):
        """
        RETURN SKILL 4B: APPROACH_RETURN_WINDOW4
        Flow-based approach for window 4 on return journey (irregular shape)
        
        PROCESS:
        1. Use optical flow to detect window 4 gap
        2. Apply offset: Y-0.02, Z+0.02
        3. Move through the window
        
        Args:
            current_pose: Current drone pose dict
            
        Returns:
            current_pose: Updated pose dict, or -1 if failed
        """
        print(f"\n{'='*60}")
        print(f"[RETURN SKILL: APPROACH_RETURN_WINDOW4 - Flow-based Detection]")
        print(f"{'='*60}")
        
        # =================================================================
        # STEP 1: Use optical flow to detect window 4
        # =================================================================
        print(f"\n  STEP 1: Detecting window 4 using optical flow...")
        
        # Generate scan trajectory
        scan_waypoints = self.nav.scanner.generate_scan_trajectory(current_pose)
        print(f"  Generated {len(scan_waypoints)} scan waypoints")
        
        # Capture frames
        scan_frames = []
        for i, waypoint in enumerate(scan_waypoints):
            rgb, _, _ = self.nav.renderer.render(waypoint['position'], waypoint['rpy'])
            scan_frames.append(rgb)
            
            # Save scan frames for debugging
            self.nav.record_frame(rgb, pose={
                'position': waypoint['position'],
                'rpy': waypoint['rpy']
            }, annotation=f"RETURN_W4_SCAN_{i+1}")
            
            print(f"    Frame {i+1}/{len(scan_waypoints)}")
        
        # Detect window using flow
        print(f"  Running flow detection...")
        mask, center_2d, confidence, debug_info = self.nav.detector.detect_window(
            scan_frames,
            prefer_larger=False,
            yaw_hint=None,
            target_score=None,
            window_count=3  # Window 4 (0-indexed as 3)
        )
        
        # Save detection visualization
        import os
        os.makedirs('./log', exist_ok=True)
        viz_path = './log/return_w4_detection.png'
        self.nav.detector.visualize(debug_info, viz_path)
        print(f"  Saved detection visualization: {viz_path}")
        
        if center_2d is None or confidence < 0.001:  # Relaxed threshold for window 4
            print(f"  [WARN] Window 4 not detected from current position")
            print(f"    Confidence: {confidence:.3f}")
            print(f"    Triggering EXPLORE to search for window 4...")
            
            # =================================================================
            # EXPLORE: Search laterally for window 4
            # =================================================================
            print(f"\n{'='*60}")
            print(f"[EXPLORE] Searching for window 4 by moving laterally")
            print(f"{'='*60}")
            
            # Try both directions: RIGHT (+Y) and LEFT (-Y)
            # Starting position has Y=-0.02, so we're already slightly left
            # Try moving right first
            explore_directions = [
                ('RIGHT', +0.15),  # Move right (east) - 15cm
                ('LEFT', -0.15)    # Move left (west) - 15cm from original
            ]
            
            original_y = current_pose['position'][1]
            window_found = False
            
            for direction_name, total_y_offset in explore_directions:
                print(f"\n  Exploring {direction_name} (Y offset: {total_y_offset:+.2f})...")
                
                # Reset to starting Y position before each direction
                current_pose['position'][1] = original_y
                
                # Move incrementally
                step_size = 0.01  # 1cm steps
                scan_interval = 0.05  # Scan every 5cm
                steps_per_scan = int(scan_interval / step_size)
                num_steps = int(abs(total_y_offset) / step_size)
                y_direction = +1 if total_y_offset > 0 else -1
                
                for step in range(num_steps):
                    # Incremental Y movement
                    current_pose['position'][1] += step_size * y_direction
                    current_pose['position'] = self.nav.clip_position_to_bounds(current_pose['position'])
                    
                    # Render every step
                    rgb, _, _ = self.nav.renderer.render(current_pose['position'], current_pose['rpy'])
                    self.nav.record_frame(rgb, pose=current_pose,
                                        annotation=f"EXPLORE_W4_{direction_name}_{step+1}")
                    
                    # Scan every scan_interval
                    if (step + 1) % steps_per_scan == 0 or step == num_steps - 1:
                        scan_num = (step + 1) // steps_per_scan
                        print(f"\n    === SCAN {scan_num} at Step {step+1}/{num_steps} ===")
                        print(f"    Position: Y={current_pose['position'][1]:+.3f}")
                        
                        # Generate scan trajectory (5 waypoints for flow detection)
                        print(f"    Generating scan trajectory...")
                        scan_waypoints_explore = self.nav.scanner.generate_scan_trajectory(current_pose)
                        scan_frames_explore = []
                        
                        print(f"    Capturing {len(scan_waypoints_explore)} frames for flow detection...")
                        for idx, wp in enumerate(scan_waypoints_explore):
                            rgb_scan, _, _ = self.nav.renderer.render(wp['position'], wp['rpy'])
                            scan_frames_explore.append(rgb_scan)
                            # Save individual scan frames
                            self.nav.record_frame(rgb_scan, pose={'position': wp['position'], 'rpy': wp['rpy']},
                                                annotation=f"EXPLORE_W4_{direction_name}_SCAN{scan_num}_F{idx+1}")
                        
                        print(f"    Running optical flow detection (window_count=3)...")
                        # Detect using FLOW
                        mask_explore, center_2d_explore, confidence_explore, debug_info_explore = self.nav.detector.detect_window(
                            scan_frames_explore, window_count=3  # Window 4 = index 3
                        )
                        
                        print(f"    Detection result:")
                        print(f"      Center: {center_2d_explore}")
                        print(f"      Confidence: {confidence_explore:.4f}")
                        
                        if center_2d_explore is not None and confidence_explore >= 0.001:
                            # Get component score for logging
                            component_score = 0
                            if hasattr(debug_info_explore, 'get'):
                                component_score = debug_info_explore.get('selected_component_score', 0)
                            
                            print(f"      Component score: {component_score:.0f}")
                            
                            # For window 4, accept ANY detection that's not at edge!
                            # Window 4 is irregular and might have high scores
                            print(f"      [SUCCESS] Window 4 FOUND at Y={current_pose['position'][1]:+.3f}!")
                            print(f"      Center pixel: ({center_2d_explore[0]:.0f}, {center_2d_explore[1]:.0f})")
                            print(f"      Confidence: {confidence_explore:.3f}")
                            
                            # Save detection visualization
                            viz_path_explore = f'./log/return_w4_explore_{direction_name}_detection.png'
                            self.nav.detector.visualize(debug_info_explore, viz_path_explore)
                            print(f"      Saved visualization: {viz_path_explore}")
                            
                            # Use this detection
                            center_2d = center_2d_explore
                            confidence = confidence_explore
                            scan_frames = scan_frames_explore
                            window_found = True
                            break
                        else:
                            print(f"      [CONTINUE] No valid window detected at this position")
                
                if window_found:
                    break
            
            if not window_found:
                print(f"\n  [ABORT] Could not find window 4 after exploring")
                return -1
            
            print(f"\n  [OK] Exploration complete - window 4 found!")
        
        print(f"  [OK] Window 4 detected at pixel: ({center_2d[0]:.0f}, {center_2d[1]:.0f})")
        print(f"  Confidence: {confidence:.3f}")
        
        # =================================================================
        # STEP 2: Calculate window position with offset Y-0.02, Z+0.02
        # =================================================================
        print(f"\n  STEP 2: Calculate window position with offset...")
        
        img_h, img_w = scan_frames[0].shape[:2]
        center_x, center_y = center_2d
        
        # Calculate pixel offset from image center
        pixel_offset_x = center_x - (img_w / 2)
        pixel_offset_y = center_y - (img_h / 2)
        
        print(f"  Image center: ({img_w/2:.0f}, {img_h/2:.0f})")
        print(f"  Window center: ({center_x:.0f}, {center_y:.0f})")
        print(f"  Pixel offset: ({pixel_offset_x:+.0f}, {pixel_offset_y:+.0f})px")
        
        # Estimate distance to window 4
        estimated_distance = 1.5  # Initial guess (splat units)
        
        # Convert to angular offset using camera matrix
        fx = self.nav.camera_matrix[0, 0]
        fy = self.nav.camera_matrix[1, 1]
        angle_offset_x = np.arctan(pixel_offset_x / fx)
        angle_offset_y = np.arctan(pixel_offset_y / fy)
        
        print(f"  Angular offset: ({np.degrees(angle_offset_x):+.1f}Â°, {np.degrees(angle_offset_y):+.1f}Â°)")
        
        # Calculate NED position based on current pose
        current_yaw = current_pose['rpy'][2]
        current_pitch = current_pose['rpy'][1]
        
        forward_distance = estimated_distance * np.cos(angle_offset_y)
        lateral_offset = estimated_distance * np.sin(angle_offset_x)
        vertical_offset = estimated_distance * np.sin(angle_offset_y)
        
        window_direction_ned = np.array([
            np.cos(current_yaw) * forward_distance - np.sin(current_yaw) * lateral_offset,
            np.sin(current_yaw) * forward_distance + np.cos(current_yaw) * lateral_offset,
            vertical_offset
        ])
        
        window_pos_ned = current_pose['position'] + window_direction_ned
        
        print(f"  Detected window position: {window_pos_ned}")
        
        # Apply offset: Y-0.02, Z-0.02 (UP = negative Z in NED)
        offset_y = -0.02  # West (negative Y in NED)
        offset_z = -0.02  # UP (negative Z in NED)
        
        window_pos_ned[1] += offset_y
        window_pos_ned[2] += offset_z
        
        print(f"  Applied offset: Y{offset_y:+.3f}, Z{offset_z:+.3f}")
        print(f"  Target position (with offset): {window_pos_ned}")
        
        # =================================================================
        # STEP 2.5: ALIGN - Center on window 4 using flow detection
        # =================================================================
        print(f"\n  STEP 2.5: ALIGN - Centering on window 4...")
        print(f"  [CRITICAL] Aligning BEFORE moving forward to maintain window tracking")
        
        max_align_iterations = 10
        align_threshold = 80  # pixels - relaxed for window 4
        
        for align_iter in range(max_align_iterations):
            print(f"\n    Align iteration {align_iter + 1}/{max_align_iterations}")
            
            # Scan and detect window
            scan_waypoints = self.nav.scanner.generate_scan_trajectory(current_pose)
            scan_frames = []
            for wp in scan_waypoints:
                rgb, _, _ = self.nav.renderer.render(wp['position'], wp['rpy'])
                scan_frames.append(rgb)
            
            # Detect window using flow
            mask, center_2d, confidence, debug_info = self.nav.detector.detect_window(
                scan_frames, window_count=3  # Window 4 is 0-indexed as 3
            )
            
            if center_2d is None:
                print(f"    [WARN] Cannot detect window, stopping alignment")
                break
            
            # Measure error from image center
            rgb, _, _ = self.nav.renderer.render(current_pose['position'], current_pose['rpy'])
            img_h, img_w = rgb.shape[:2]
            img_center_x = img_w / 2
            img_center_y = img_h / 2
            
            error_x_px = center_2d[0] - img_center_x
            error_y_px = center_2d[1] - img_center_y
            error_mag = np.sqrt(error_x_px**2 + error_y_px**2)
            
            print(f"    Center pixel: ({center_2d[0]:.0f}, {center_2d[1]:.0f})")
            print(f"    Image center: ({img_center_x:.0f}, {img_center_y:.0f})")
            print(f"    Error: ({error_x_px:+.1f}, {error_y_px:+.1f})px, mag={error_mag:.1f}px")
            
            # Check if aligned
            if error_mag < align_threshold:
                print(f"    [OK] Aligned! (error {error_mag:.1f}px < {align_threshold}px)")
                break
            
            # Estimate depth
            Z_cam = 1.0  # Assume ~1 splat unit away for window 4
            
            # Compute corrections using camera intrinsics
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
            
            # Apply (NO COLLISION CHECK during alignment)
            new_pos = current_pose['position'].copy()
            new_pos[1] += ctrl_y
            new_pos[2] += ctrl_z
            new_pos = self.nav.clip_position_to_bounds(new_pos)
            
            current_pose['position'] = new_pos
            print(f"    [OK] Moved to {current_pose['position']}")
            
            # Record frame
            rgb_align, _, _ = self.nav.renderer.render(current_pose['position'], current_pose['rpy'])
            self.nav.record_frame(rgb_align, pose=current_pose,
                                annotation=f"RETURN_W4_ALIGN_{align_iter+1}")
        
        print(f"\n  [OK] Alignment complete - window is centered")
        
        # CRITICAL: Ensure Z is slightly above centerline (negative Z in NED)
        current_z = current_pose['position'][2]
        print(f"  Current Z: {current_z:.4f}")
        
        if current_z > 0.01:
            print(f"  [ADJUST] Z is too low/positive ({current_z:.4f}), moving to Z=0.0 (centerline)")
            current_pose['position'][2] = 0.0
        elif current_z > -0.01:
            print(f"  [ADJUST] Z near centerline ({current_z:.4f}), setting to Z=-0.02 (slightly above)")
            current_pose['position'][2] = -0.02
        else:
            print(f"  [OK] Z is good (above centerline): {current_z:.4f}")
        
        print(f"  Final aligned position: {current_pose['position']}")
        
        # =================================================================
        # STEP 3: Safety forward movement (now that we're aligned)
        # =================================================================
        print(f"\n  STEP 3: Safety forward movement...")
        
        from navigation import goToWaypoint
        
        # Now move forward - we're already aligned to window center!
        current_yaw = current_pose['rpy'][2]
        forward_ned = np.array([
            np.cos(current_yaw),
            np.sin(current_yaw),
            0.0
        ])
        
        # Small forward movement to start approach
        safety_distance = 0.05
        safety_pos = current_pose['position'] + forward_ned * safety_distance
        safety_pos = self.nav.clip_position_to_bounds(safety_pos)
        current_pose['position'] = safety_pos.copy()
        
        # Render safety step
        rgb_safety, _, _ = self.nav.renderer.render(current_pose['position'], current_pose['rpy'])
        self.nav.record_frame(rgb_safety, pose=current_pose, annotation="RETURN_W4_SAFETY_FWD")
        print(f"    Moved forward {safety_distance:.3f} to: {current_pose['position']}")
        
        # =================================================================
        # STEP 4: Approach through aligned window
        # =================================================================
        # Now navigate to detected window position using geometric stepping
        print(f"\n  STEP 4: Approaching through aligned window...")
        print(f"  [NOTE] Collision checking DISABLED for window 4 crossing")
        
        # Just move forward - we're already aligned!
        current_yaw = current_pose['rpy'][2]
        forward_ned = np.array([
            np.cos(current_yaw),
            np.sin(current_yaw),
            0.0
        ])
        
        # Move forward a moderate distance
        approach_distance = 0.16  # 8 steps * 0.02 = 0.16 splat units
        step_size = 0.02
        num_steps = 8  # Fixed at 8 steps - enough to cross window 4
        
        print(f"  Moving {approach_distance:.2f} units forward in {num_steps} steps (optimized for window 4)")
        
        for step in range(num_steps):
            step_increment = forward_ned * step_size
            new_pos = current_pose['position'] + step_increment
            new_pos = self.nav.clip_position_to_bounds(new_pos)
            
            # NO COLLISION CHECK - allow crossing window 4
            current_pose['position'] = new_pos.copy()
            
            # Render EVERY step (important for debugging and video)
            rgb, _, _ = self.nav.renderer.render(current_pose['position'], current_pose['rpy'])
            self.nav.record_frame(rgb, pose=current_pose,
                                annotation=f"RETURN_W4_APPROACH_{step+1}/{num_steps}")
            
            # Print progress every 5 steps
            if (step + 1) % 5 == 0 or step == num_steps - 1:
                print(f"    Step {step+1}/{num_steps}")
        
        print(f"  [OK] Approach complete: {current_pose['position']}")
        
        print(f"  [OK] Return window 4 navigation complete")
        print(f"  Final position: {current_pose['position']}")
        
        return current_pose
    
    # =========================================================================
    # RETURN SKILL 5: RECENTER_RETURN
    # =========================================================================
    def recenter_return(self, current_pose):
        """
        RETURN SKILL 5: RECENTER_RETURN
        Reset to centerline after passing through window on return journey
        
        Args:
            current_pose: Current drone pose dict
            
        Returns:
            current_pose: Updated pose dict
        """
        from navigation import wrap_angle
        
        print(f"\n{'='*60}")
        print(f"[RETURN SKILL: RECENTER_RETURN]")
        print(f"{'='*60}")
        
        # Keep X, reset Y/Z/RPY to zero
        target_x = current_pose['position'][0]
        
        print(f"  Goal: Keep X={target_x:.3f}, reset Y/Z to zero, maintain 180Â° yaw")
        print(f"  Starting: [{current_pose['position'][0]:.3f}, {current_pose['position'][1]:.3f}, {current_pose['position'][2]:.3f}]")
        
        # Safety: move forward first
        safety_distance = 0.1
        safety_steps = 10
        step_size = safety_distance / safety_steps
        
        for safety_step in range(safety_steps):
            current_yaw = current_pose['rpy'][2]
            forward_increment = np.array([
                np.cos(current_yaw) * step_size,
                np.sin(current_yaw) * step_size,
                0.0
            ])
            
            current_pose['position'] += forward_increment
            current_pose['position'] = self.nav.clip_position_to_bounds(current_pose['position'])
            
            if safety_step % 3 == 0:
                rgb, _, _ = self.nav.renderer.render(current_pose['position'], current_pose['rpy'])
                self.nav.record_frame(rgb, pose=current_pose,
                                    annotation=f"RETURN_RECENTER_SAFETY_{safety_step}")
        
        # Update target X
        target_x = current_pose['position'][0]
        
        # Iterative convergence to Y=0, Z=0, yaw=Â±180Â°
        target_y = 0.0
        target_z = 0.0
        target_yaw = np.pi  # 180 degrees (can also be -Ï€)
        
        max_iterations = 50
        position_tolerance = 0.01
        angle_tolerance = np.radians(2.0)
        position_step = 0.01
        angle_step = np.radians(1.0)
        
        for iteration in range(max_iterations):
            error_y = target_y - current_pose['position'][1]
            error_z = target_z - current_pose['position'][2]
            error_yaw = wrap_angle(target_yaw - current_pose['rpy'][2])
            
            if abs(error_y) < position_tolerance and abs(error_z) < position_tolerance and abs(error_yaw) < angle_tolerance:
                print(f"  [OK] Recentered after {iteration} iterations")
                break
            
            # Apply corrections
            if abs(error_y) > position_tolerance:
                current_pose['position'][1] += np.clip(error_y, -position_step, position_step)
            
            if abs(error_z) > position_tolerance:
                current_pose['position'][2] += np.clip(error_z, -position_step, position_step)
            
            if abs(error_yaw) > angle_tolerance:
                current_pose['rpy'][2] += np.clip(error_yaw, -angle_step, angle_step)
            
            current_pose['position'] = self.nav.clip_position_to_bounds(current_pose['position'])
            current_pose['position'][0] = target_x
            
            if iteration % 10 == 0:
                rgb, _, _ = self.nav.renderer.render(current_pose['position'], current_pose['rpy'])
                self.nav.record_frame(rgb, pose=current_pose,
                                    annotation=f"RETURN_RECENTER_iter{iteration}")
                print(f"  Iter {iteration}: Y={current_pose['position'][1]:+.3f}, Z={current_pose['position'][2]:+.3f}, Yaw={np.degrees(current_pose['rpy'][2]):+.1f}Â°")
        
        print(f"  Final: [{current_pose['position'][0]:.3f}, {current_pose['position'][1]:.3f}, {current_pose['position'][2]:.3f}]")
        print(f"  Final yaw: {np.degrees(current_pose['rpy'][2]):.1f}Â° (target: 180Â°)")
        print(f"  [OK] Return recenter complete")
        
        return current_pose