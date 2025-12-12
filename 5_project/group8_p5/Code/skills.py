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
        
        # Re-scan to get fresh detection
        # Pass yaw_hint for directional selection: +ve = RIGHT, -ve = LEFT
        verified_window, corners_2d, scan_data = self.scan(
            current_pose, scan_type=f'verify_c{cycle_num}', yaw_hint=self.yaw_error_initial
        )
        
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
        Move through the aligned window using trajectory following
        
        Args:
            current_pose: Current drone pose dict (must be well-aligned!)
            window_3d_pos: Window position
            
        Returns:
            current_pose: Updated pose dict, or -1 if failed
        """
        print(f"\n{'='*60}")
        print(f"[SKILL: APPROACH]")
        print(f"{'='*60}")
        
        # CRITICAL FIX: Don't use zero_rpy - it doesn't mean "level" in renderer coords!
        # The renderer applies init_orientation transform, so zero angles = TILTED view
        # Instead, maintain CURRENT orientation which is already well-aligned
        approach_rpy = current_pose['rpy'].copy()
        
        print(f"  Maintaining current orientation for approach:")
        print(f"    roll={np.degrees(approach_rpy[0]):.1f}°, "
              f"pitch={np.degrees(approach_rpy[1]):.1f}°, yaw={np.degrees(approach_rpy[2]):.1f}°")
        print(f"  (Renderer will transform this to maintain level view)")
        
        # Log alignment for debugging
        proj_pixel = self.nav.pnp_estimator.project_window_to_pixel(window_3d_pos, current_pose)
        
        if proj_pixel is not None:
            rgb, _, _ = self.nav.renderer.render(current_pose['position'], current_pose['rpy'])
            img_h, img_w = rgb.shape[:2]
            error_x = proj_pixel[0] - img_w / 2
            error_y = proj_pixel[1] - img_h / 2
            error_mag = np.sqrt(error_x**2 + error_y**2)
            
            print(f"  Pre-approach pixel error: {error_mag:.1f}px")
        
        # Calculate distance and direction
        distance = np.linalg.norm(window_3d_pos - current_pose['position'])
        print(f"  Distance to window: {distance:.3f} splat units")
        
        # CRITICAL FIX: Move in LOCAL body frame (forward = +X in body frame)
        # Not global NED! The drone should move FORWARD relative to its orientation.
        
        # Get current yaw from current_pose
        current_yaw = current_pose['rpy'][2]
        
        # Create rotation matrix for yaw (NED to body frame)
        # In NED: X=North, Y=East, Z=Down
        # Body frame: X=Forward, Y=Right, Z=Down
        cos_yaw = np.cos(current_yaw)
        sin_yaw = np.sin(current_yaw)
        
        # Rotation matrix: NED -> Body
        R_ned_to_body = np.array([
            [ cos_yaw, sin_yaw, 0],  # Body X = cos(yaw)*NED_X + sin(yaw)*NED_Y
            [-sin_yaw, cos_yaw, 0],  # Body Y = -sin(yaw)*NED_X + cos(yaw)*NED_Y
            [ 0,       0,       1]   # Body Z = NED_Z
        ])
        
        # Direction to window in NED
        direction_ned = window_3d_pos - current_pose['position']
        
        # Transform to body frame
        direction_body = R_ned_to_body @ direction_ned
        
        print(f"  Direction in NED: {direction_ned}")
        print(f"  Direction in Body: {direction_body}")
        print(f"    (Body X=forward, Y=right, Z=down)")
        
        # We want to move FORWARD (body +X), so create forward unit vector in body frame
        forward_body = np.array([1.0, 0.0, 0.0])  # Pure forward
        
        # Transform back to NED for actual movement
        R_body_to_ned = R_ned_to_body.T  # Inverse rotation
        forward_ned = R_body_to_ned @ forward_body
        
        print(f"  Forward direction in NED: {forward_ned}")
        print(f"  (This is the direction the drone will move)")
        
        # Normalize
        forward_ned_norm = forward_ned / np.linalg.norm(forward_ned)
        
        # Use forward direction for movement
        direction_to_use = forward_ned_norm
        
        # Move in steps toward window
        step_size = 0.1  # Small steps for safety
        num_steps = int(np.ceil(distance / step_size))
        num_steps = max(3, min(num_steps, 10))  # 3-10 steps
        
        print(f"  Moving in {num_steps} steps (direct position updates)")
        
        for step in range(num_steps):
            # Calculate target for this step (moving FORWARD in body frame)
            progress = (step + 1) / num_steps
            target_pos = current_pose['position'] + direction_to_use * (progress * distance)
            target_pos = self.nav.clip_position_to_bounds(target_pos)
            
            # Collision check
            if doesItCollide(target_pos):
                print(f"  [ERROR] Step {step+1} would collide at {target_pos}")
                return -1
            
            print(f"  Step {step+1}/{num_steps}: Moving to {target_pos}")
            
            # DIRECT POSITION UPDATE (no dynamics, no tilting!)
            current_pose['position'] = target_pos.copy()
            
            # DEBUG: Log what we're about to render
            print(f"    [DEBUG] About to render:")
            print(f"      Position (NED): {current_pose['position']}")
            print(f"      Orientation being used: {approach_rpy} (roll={np.degrees(approach_rpy[0]):.1f}°, "
                  f"pitch={np.degrees(approach_rpy[1]):.1f}°, yaw={np.degrees(approach_rpy[2]):.1f}°)")
            
            # Render frame with CURRENT orientation (maintains level view)
            rgb, _, _ = self.nav.renderer.render(current_pose['position'], approach_rpy)
            
            # DEBUG: Verify what orientation is in the pose we're recording
            debug_pose = {'position': current_pose['position'], 'rpy': approach_rpy}
            print(f"    [DEBUG] Recording frame with pose: pos={debug_pose['position']}, "
                  f"rpy={np.degrees(debug_pose['rpy'])}°")
            
            self.nav.record_frame(rgb, pose=debug_pose,
                                annotation=f"APPROACH_W{self.nav.window_count}_S{step+1}/{num_steps}")
        
        print(f"  [OK] Approach complete")
        print(f"  Final position (NED): {current_pose['position']}")
        
        return current_pose
    
    # =========================================================================
    # SKILL 6: RECENTER
    # =========================================================================
    def recenter(self, current_pose):
        """
        SKILL 6: RECENTER
        Reset yaw to 0° only (skip Y/Z to avoid collisions)
        
        Args:
            current_pose: Current drone pose dict
            
        Returns:
            current_pose: Updated pose dict
        """
        print(f"\n{'='*60}")
        print(f"[SKILL: RECENTER]")
        print(f"{'='*60}")
        
        current_yaw_deg = np.degrees(current_pose['rpy'][2])
        
        if abs(current_yaw_deg) > 2.0:
            print(f"  Resetting yaw from {current_yaw_deg:.1f}° to 0°")
            from navigation import goToWaypoint_yaw
            result = goToWaypoint_yaw(
                current_pose, 0.0,
                pose_history=self.nav.pose_history,
                action=f'RECENTER_YAW_W{self.nav.window_count}',
                navigator=self.nav
            )
            
            if result != -1:
                current_pose = result
        else:
            print(f"  Yaw already near 0° ({current_yaw_deg:.1f}°), skipping")
        
        # Skip Y/Z reset - causes collisions
        print(f"  Position: [{current_pose['position'][0]:.2f}, "
              f"{current_pose['position'][1]:.2f}, {current_pose['position'][2]:.2f}]")
        print(f"  [OK] Recentered (yaw only)")
        
        return current_pose