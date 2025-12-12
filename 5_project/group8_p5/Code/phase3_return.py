"""
Phase 3: Return Journey
Navigate back through windows 4â†’3â†’2â†’1
"""

import numpy as np
from returnskills import ReturnNavigationSkills
from navigation import goToWaypoint, goToWaypoint_yaw, wrap_angle


def run_return_journey(navigator, renderer, currentPose):
    """
    Execute return journey through windows 4â†’3â†’2â†’1
    
    Args:
        navigator: WindowNavigator instance
        renderer: SplatRenderer instance
        currentPose: Current drone pose dict (should be facing 180Â° after turnback)
        
    Returns:
        currentPose: Updated pose after completing return journey, or -1 on failure
    """
    print("\n[RETURN JOURNEY] Starting navigation through windows 4â†’1")
    print(f"Current position: {currentPose['position']}")
    print(f"Current yaw: {np.degrees(currentPose['rpy'][2]):.1f}Â°")
    
    # ==========================================================================
    # STEP 1: Navigate to starting position [1.75, -0.02, 0.0] with yaw=180Â°
    # ==========================================================================
    print(f"\n{'='*70}")
    print(f"RETURN PREP: Moving to start position [1.75, -0.02, 0.0]")
    print(f"{'='*70}")
    
    start_position = np.array([1.75, -0.02, 0.0])
    print(f"  Target position: {start_position}")
    print(f"  Current position: {currentPose['position']}")
    
    # Check if we're already at the start position
    distance_to_start = np.linalg.norm(currentPose['position'] - start_position)
    print(f"  Distance to start: {distance_to_start:.3f} units")
    
    if distance_to_start > 0.05:  # More than 5cm away
        print(f"  Navigating to start position...")
        
        # Navigate to start position
        result = goToWaypoint(
            currentPose, start_position,
            velocity=0.05,
            pose_history=navigator.pose_history,
            action='RETURN_START_POS',
            lock_roll_pitch=True,
            navigator=navigator
        )
        
        if result == -1:
            print(f"  [ERROR] Failed to reach start position")
            return -1
        
        currentPose = result
        print(f"  [OK] Reached start position: {currentPose['position']}")
    else:
        print(f"  [OK] Already at start position (within 5cm)")
    
    # Ensure yaw is 180Â°
    target_yaw = np.pi  # 180 degrees
    current_yaw_error = abs(wrap_angle(target_yaw - currentPose['rpy'][2]))
    
    if current_yaw_error > np.radians(5.0):  # More than 5Â° off
        print(f"\n  Adjusting yaw to 180Â°...")
        print(f"    Current yaw: {np.degrees(currentPose['rpy'][2]):.1f}Â°")
        print(f"    Target yaw: 180.0Â°")
        
        result = goToWaypoint_yaw(
            currentPose, target_yaw,
            pose_history=navigator.pose_history,
            action='RETURN_YAW_ALIGN',
            navigator=navigator
        )
        
        if result == -1:
            print(f"  [ERROR] Failed to align yaw")
            return -1
        
        currentPose = result
        print(f"  [OK] Yaw aligned: {np.degrees(currentPose['rpy'][2]):.1f}Â°")
    else:
        print(f"  [OK] Yaw already aligned: {np.degrees(currentPose['rpy'][2]):.1f}Â°")
    
    print(f"\n  [OK] Ready to start return navigation")
    print(f"  Position: {currentPose['position']}")
    print(f"  Yaw: {np.degrees(currentPose['rpy'][2]):.1f}Â°")
    
    # ==========================================================================
    # STEP 2: Begin return skills
    # ==========================================================================
    # Initialize return skills
    return_skills = ReturnNavigationSkills(navigator)
    
    # CRITICAL: Reset window_count for return journey
    # Window 3 on return = index 2 (use PnP), not index 4 (flow)
    # We'll manually set window_count before each window
    
    # Return through 4 windows in reverse order
    return_window_order = [4, 3, 2, 1]
    
    for return_win_num in return_window_order:
        print(f"\n{'='*70}")
        print(f"RETURN WINDOW {return_win_num} / 4")
        print(f"{'='*70}")
        
        # Set navigator.window_count to match current window (0-indexed)
        # Window 4 → index 3 (flow), Window 3 → index 2 (PnP), etc.
        navigator.window_count = return_win_num - 1
        print(f"  Set navigator.window_count = {navigator.window_count}")
        
        # =================================================================
        # WINDOW 4: Special flow-based handling (irregular shape)
        # =================================================================
        if return_win_num == 4:
            print(f"[RETURN WINDOW 4] Flow-based navigation (irregular shape)")
            
            # Direct approach - no scan/align needed (already positioned)
            result = return_skills.approach_return_window4(currentPose)
            
            if result == -1:
                print(f"\n[ABORT] Return window 4 approach failed")
                return -1
            
            currentPose = result
            
            # Recenter after passing through
            currentPose = return_skills.recenter_return(currentPose)
            
            print(f"\n[OK] Return window 4 complete!")
            return_skills.return_window_count += 1
            
        # =================================================================
        # WINDOWS 3, 2, 1: Normal PnP-based navigation
        # =================================================================
        else:
            print(f"[RETURN WINDOW {return_win_num}] PnP-based navigation")
            
            # SCAN
            window_3d_pos, corners_2d, scan_data = return_skills.scan_return(
                currentPose, scan_type=f'return_w{return_win_num}_initial'
            )
            
            if window_3d_pos is None:
                print(f"\n[ABORT] Could not detect return window {return_win_num}")
                return -1
            
            # =================================================================
            # ALIGN-VERIFY LOOP (BEFORE YAW CORRECTION!)
            # =================================================================
            print(f"\n{'='*70}")
            print(f"[RETURN ALIGN-VERIFY - Pre-Yaw Correction]")
            print(f"{'='*70}")
            print(f"Strategy: Align position FIRST, then rotate yaw to maintain tracking")
            
            # Check current alignment
            proj_pixel = navigator.pnp_estimator.project_window_to_pixel(window_3d_pos, currentPose)
            
            if proj_pixel is not None:
                img_h, img_w = renderer.image_height, renderer.image_width
                error_x = proj_pixel[0] - img_w / 2
                error_y = proj_pixel[1] - img_h / 2
                error_mag = np.sqrt(error_x**2 + error_y**2)
                
                print(f"Initial alignment error: {error_mag:.1f}px")
                
                if error_mag < 25:
                    print(f"  Good alignment ({error_mag:.1f}px < 25px)")
                    print(f"  Proceeding to yaw correction")
                else:
                    print(f"  Needs refinement ({error_mag:.1f}px > 25px)")
                    print(f"  Running RETURN ALIGN-VERIFY cycles")
                    
                    # ITERATIVE ALIGN-VERIFY LOOP
                    max_cycles = 3
                    for cycle_num in range(max_cycles):
                        print(f"\n  --- Cycle {cycle_num + 1}/{max_cycles} ---")
                        
                        # ALIGN
                        currentPose, error_before = return_skills.align_return(
                            currentPose, window_3d_pos, corners_2d
                        )
                        
                        # VERIFY - Re-scan to confirm window position
                        print(f"  [VERIFY] Re-scanning window...")
                        verified_window, verified_corners, scan_data = return_skills.scan_return(
                            currentPose, scan_type=f'return_w{return_win_num}_verify_c{cycle_num}'
                        )
                        
                        if verified_window is None:
                            print(f"  [WARN] Verification failed")
                            print(f"  [RETRY] Moving backward...")
                            
                            # Move backward (for return journey with yaw~180°, backward = +X)
                            step_size = 0.02
                            new_pos = currentPose['position'].copy()
                            new_pos[0] += step_size
                            new_pos = navigator.clip_position_to_bounds(new_pos)
                            currentPose['position'] = new_pos
                            
                            # Retry verification
                            verified_window, verified_corners, scan_data = return_skills.scan_return(
                                currentPose, scan_type=f'return_w{return_win_num}_verify_c{cycle_num}_retry'
                            )
                            
                            if verified_window is None:
                                print(f"  [WARN] Second verify failed, continuing")
                                break
                            else:
                                window_3d_pos = verified_window
                                corners_2d = verified_corners
                        else:
                            window_3d_pos = verified_window
                            corners_2d = verified_corners
                        
                        # Check alignment after verify
                        proj_pixel_check = navigator.pnp_estimator.project_window_to_pixel(
                            window_3d_pos, currentPose
                        )
                        
                        if proj_pixel_check is not None:
                            error_x_check = proj_pixel_check[0] - img_w / 2
                            error_y_check = proj_pixel_check[1] - img_h / 2
                            error_mag = np.sqrt(error_x_check**2 + error_y_check**2)
                        else:
                            error_mag = float('inf')
                        
                        alignment_good = error_mag < 50
                        
                        # Check convergence
                        if alignment_good or error_mag < 20:
                            print(f"  [CONVERGED] error={error_mag:.1f}px")
                            break
                        elif cycle_num < max_cycles - 1:
                            print(f"  [CONTINUE] error={error_mag:.1f}px")
                        else:
                            print(f"  [MAX CYCLES] Proceeding anyway")
            else:
                print(f"  [WARN] Cannot project window")
            
            # =================================================================
            # FIX_YAW (AFTER ALIGNMENT)
            # =================================================================
            print(f"\n{'='*70}")
            print(f"[FIX_YAW - After Position Alignment]")
            print(f"{'='*70}")
            
            result = return_skills.fix_yaw_return(currentPose, window_3d_pos)
            if result == -1:
                print(f"\n[ABORT] Return yaw alignment failed")
                return -1
            currentPose = result
            
            # =================================================================
            # FINAL VERIFICATION (After Yaw Correction)
            # =================================================================
            print(f"\n{'='*70}")
            print(f"[FINAL VERIFICATION - Post-Yaw]")
            print(f"{'='*70}")
            
            # Quick final check
            proj_pixel_final = navigator.pnp_estimator.project_window_to_pixel(window_3d_pos, currentPose)
            
            if proj_pixel_final is not None:
                error_x_final = proj_pixel_final[0] - img_w / 2
                error_y_final = proj_pixel_final[1] - img_h / 2
                error_mag_final = np.sqrt(error_x_final**2 + error_y_final**2)
                
                print(f"Final alignment error: {error_mag_final:.1f}px")
                
                if error_mag_final > 50:
                    print(f"  [WARN] Alignment degraded after yaw correction")
                    print(f"  Running one more ALIGN pass...")
                    currentPose, _ = return_skills.align_return(currentPose, window_3d_pos, corners_2d)
            
            # =================================================================
            # APPROACH
            # =================================================================
            result = return_skills.approach_return(currentPose, window_3d_pos)
            if result == -1:
                print(f"\n[ABORT] Return approach failed")
                return -1
            currentPose = result
            
            # RECENTER
            currentPose = return_skills.recenter_return(currentPose)
            
            print(f"\n[OK] Return window {return_win_num} complete!")
            return_skills.return_window_count += 1
    
    # =========================================================================
    # RETURN JOURNEY COMPLETE
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"RETURN JOURNEY COMPLETE")
    print(f"{'='*70}")
    print(f"  Windows traversed: {return_skills.return_window_count}")
    print(f"  Final position: {currentPose['position']}")
    print(f"  Final yaw: {np.degrees(currentPose['rpy'][2]):.1f}Â°")
    print(f"{'='*70}")
    
    return currentPose