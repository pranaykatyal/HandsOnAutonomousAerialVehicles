"""
Return Journey Navigation Skills

Skills for navigating back through windows 4→3→2→1 after turning around.
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
        self.return_window_count = 0  # Track windows on return journey (4→3→2→1)
        
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
                print(f"  Yaw iter {yaw_iter}: Yaw={np.degrees(current_pose['rpy'][2]):.1f}°, "
                      f"Error={np.degrees(yaw_error):.1f}°")
            
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
        Uses tracked position for distance-based passing detection
        
        Args:
            current_pose: Current drone pose dict
            
        Returns:
            current_pose: Updated pose dict, or -1 if failed
        """
        print(f"\n{'='*60}")
        print(f"[RETURN SKILL: APPROACH_RETURN_WINDOW4 - Flow-based]")
        print(f"{'='*60}")
        
        # Get forward direction (now facing 180°)
        current_yaw = current_pose['rpy'][2]
        forward_ned = np.array([
            np.cos(current_yaw),
            np.sin(current_yaw),
            0.0
        ])
        
        print(f"  Current yaw: {np.degrees(current_yaw):.1f} deg")
        print(f"  Forward direction (NED): {forward_ned}")
        
        # Approach parameters for window 4
        step_size = 0.02
        max_steps = 20  # Same as forward journey
        
        # Apply offset for window 4 before approach (same as forward)
        print(f"  Applying window 4 offset: Y=+0.02, Z=+0.02")
        current_pose['position'][1] += 0.02  # East
        current_pose['position'][2] += 0.02  # Down
        current_pose['position'] = self.nav.clip_position_to_bounds(current_pose['position'])
        print(f"  New position after offset: {current_pose['position']}")
        
        print(f"  Moving forward (return through window 4)")
        print(f"  Max steps: {max_steps}, Step size: {step_size:.3f}")
        
        for step in range(max_steps):
            # Move forward
            step_increment = forward_ned * step_size
            new_pos = current_pose['position'] + step_increment
            new_pos = self.nav.clip_position_to_bounds(new_pos)
            current_pose['position'] = new_pos.copy()
            
            # Render
            rgb, _, _ = self.nav.renderer.render(current_pose['position'], current_pose['rpy'])
            self.nav.record_frame(rgb, pose=current_pose,
                                annotation=f"RETURN_W4_STEP_{step+1}/{max_steps}")
            
            if (step + 1) % 5 == 0:
                print(f"  Step {step+1}/{max_steps}")
        
        print(f"  [OK] Return window 4 approach complete")
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
        
        print(f"  Goal: Keep X={target_x:.3f}, reset Y/Z to zero, maintain 180° yaw")
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
        
        # Iterative convergence to Y=0, Z=0, yaw=±180°
        target_y = 0.0
        target_z = 0.0
        target_yaw = np.pi  # 180 degrees (can also be -π)
        
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
                print(f"  Iter {iteration}: Y={current_pose['position'][1]:+.3f}, Z={current_pose['position'][2]:+.3f}, Yaw={np.degrees(current_pose['rpy'][2]):+.1f}°")
        
        print(f"  Final: [{current_pose['position'][0]:.3f}, {current_pose['position'][1]:.3f}, {current_pose['position'][2]:.3f}]")
        print(f"  Final yaw: {np.degrees(current_pose['rpy'][2]):.1f}° (target: 180°)")
        print(f"  [OK] Return recenter complete")
        
        return current_pose