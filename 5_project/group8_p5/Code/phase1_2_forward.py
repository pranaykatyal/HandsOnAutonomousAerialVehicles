"""
Phase 1&2: Forward Journey
Navigate through windows 1→2→3→4
"""

import numpy as np
from skills import NavigationSkills
from navigation import wrap_angle


def run_forward_journey(navigator, renderer, currentPose):
    """
    Execute forward journey through windows 1→2→3→4
    
    Args:
        navigator: WindowNavigator instance
        renderer: SplatRenderer instance  
        currentPose: Current drone pose dict with 'position' and 'rpy'
        
    Returns:
        currentPose: Updated pose after completing forward journey, or -1 on failure
    """
    print("\n[FORWARD JOURNEY] Starting navigation through windows 1→4")
    
    # Initialize skills
    skills = NavigationSkills(navigator)
    
    max_windows = 4
    
    for window_num in range(max_windows):
        print(f"\n{'='*70}")
        print(f"WINDOW {window_num + 1} / {max_windows}")
        print(f"{'='*70}")
        print(f"Flow: SCAN → FIX_YAW → ALIGN → VERIFY → APPROACH → RECENTER")
        
        # =================================================================
        # SKILL 1: SCAN
        # =================================================================
        window_3d_pos, corners_2d, scan_data = skills.scan(currentPose, scan_type='initial')
        
        if window_3d_pos is None:
            print(f"\n[ABORT] Could not detect window {window_num + 1}")
            return -1
        
        # =================================================================
        # SKILL 2: FIX_YAW (Conditional - skip for window 4)
        # =================================================================
        if navigator.window_count >= 3:
            print(f"\n[WINDOW 4] Skipping FIX_YAW - irregular shape")
            skills.yaw_error_initial = 0.0
        else:
            # Check if yaw alignment is needed
            vec_to_window = window_3d_pos - currentPose['position']
            desired_yaw = np.arctan2(vec_to_window[1], vec_to_window[0])
            current_yaw = currentPose['rpy'][2]
            yaw_error = wrap_angle(desired_yaw - current_yaw)
            yaw_error_deg = np.degrees(abs(yaw_error))
            
            # Store yaw error for VERIFY guidance
            skills.yaw_error_initial = yaw_error
            
            print(f"\nYaw check: current={np.degrees(current_yaw):.1f}°, "
                  f"desired={np.degrees(desired_yaw):.1f}°, error={yaw_error_deg:.1f}°")
            print(f"  Yaw hint: {np.degrees(yaw_error):.1f}° ({'RIGHT' if yaw_error > 0 else 'LEFT'})")
            
            if yaw_error_deg > 5.0:
                print(f"  Running FIX_YAW (error > 5°)")
                result = skills.fix_yaw(currentPose, window_3d_pos)
                
                if result == -1:
                    print(f"\n[ABORT] Yaw alignment failed")
                    return -1
                
                currentPose = result
            else:
                print(f"  Skipping FIX_YAW (error < 5°)")
        
        # =================================================================
        # SKILLS 3-4: ALIGN-VERIFY LOOP (Conditional)
        # =================================================================
        print(f"\n{'='*70}")
        print(f"[ALIGN-VERIFY CHECK]")
        print(f"{'='*70}")
        
        if navigator.window_count >= 3:
            # Window 4: Single alignment only
            print(f"[WINDOW 4] Single alignment pass")
            
            currentPose, error_mag = skills.align(currentPose, window_3d_pos, corners_2d)
            print(f"  Alignment error: {error_mag:.1f}px")
            print(f"  Proceeding to APPROACH")
        else:
            # Windows 1-3: Check if alignment needed
            proj_pixel = navigator.pnp_estimator.project_window_to_pixel(window_3d_pos, currentPose)
            
            if proj_pixel is not None:
                img_h, img_w = renderer.image_height, renderer.image_width
                error_x = proj_pixel[0] - img_w / 2
                error_y = proj_pixel[1] - img_h / 2
                error_mag = np.sqrt(error_x**2 + error_y**2)
                
                print(f"Post-yaw error: {error_mag:.1f}px")
                
                if error_mag < 25:
                    print(f"  Excellent alignment ({error_mag:.1f}px < 25px)")
                    print(f"  Skipping ALIGN-VERIFY")
                else:
                    print(f"  Needs refinement ({error_mag:.1f}px > 25px)")
                    print(f"  Running ALIGN-VERIFY cycles")
                    
                    # ITERATIVE ALIGN-VERIFY LOOP
                    max_cycles = 3
                    for cycle_num in range(max_cycles):
                        print(f"\n  --- Cycle {cycle_num + 1}/{max_cycles} ---")
                        
                        # ALIGN
                        currentPose, error_before = skills.align(currentPose, window_3d_pos, corners_2d)
                        
                        # VERIFY
                        verified_window, corners_2d, alignment_good, error_mag = skills.verify(
                            currentPose, window_3d_pos, cycle_num=cycle_num
                        )
                        
                        if verified_window is None:
                            print(f"  [WARN] Verification failed")
                            print(f"  [RETRY] Moving backward...")
                            
                            # Move backward
                            step_size = 0.02
                            new_pos = currentPose['position'].copy()
                            new_pos[0] -= step_size
                            new_pos = navigator.clip_position_to_bounds(new_pos)
                            currentPose['position'] = new_pos
                            
                            # Retry
                            verified_window, corners_2d, alignment_good, error_mag = skills.verify(
                                currentPose, window_3d_pos, cycle_num=cycle_num
                            )
                            
                            if verified_window is None:
                                print(f"  [WARN] Second verify failed, continuing")
                                break
                            else:
                                window_3d_pos = verified_window
                        else:
                            window_3d_pos = verified_window
                        
                        # Check convergence
                        if alignment_good or error_mag < 20:
                            print(f"  [CONVERGED] error={error_mag:.1f}px")
                            break
                        elif cycle_num < max_cycles - 1:
                            print(f"  [CONTINUE] error={error_mag:.1f}px")
                        else:
                            print(f"  [MAX CYCLES] Proceeding anyway")
            else:
                print(f"  [WARN] Cannot project - skipping ALIGN-VERIFY")
        
        # =================================================================
        # SKILL 5: APPROACH
        # =================================================================
        start_pos = currentPose['position'].copy()
        
        result = skills.approach(currentPose, window_3d_pos)
        
        if result == -1:
            print(f"\n[ABORT] Approach failed")
            return -1
        
        currentPose = result
        navigator.window_count += 1
        
        # =================================================================
        # SKILL 6: RECENTER
        # =================================================================
        y_displacement = currentPose['position'][1] - start_pos[1]
        came_from_left = y_displacement < 0
        
        print(f"\n  Y displacement: {y_displacement:+.3f}")
        print(f"  Came from: {'LEFT' if came_from_left else 'RIGHT'}")
        
        currentPose = skills.recenter(currentPose)
        
        # =================================================================
        # SKILL 7: TURNBACK (After window 4) or EXPLORE
        # =================================================================
        if navigator.window_count >= 4:
            # Turn back after window 4
            print(f"\n{'='*70}")
            print(f"[WINDOW 4 COMPLETE - TURNING BACK 180°]")
            print(f"{'='*70}")
            
            currentPose = skills.turnback(currentPose)
            
            # Recenter with 180° yaw
            from returnskills import ReturnNavigationSkills
            return_skills = ReturnNavigationSkills(navigator)
            currentPose = return_skills.recenter_return(currentPose)
            
            print(f"\n[OK] Window 4 complete! Ready for return journey.")
            break  # Exit forward journey loop
        else:
            # Check for next window
            print(f"\n{'='*70}")
            print(f"[CHECKING FOR NEXT WINDOW]")
            print(f"{'='*70}")
            
            next_window, _, _ = skills.scan(currentPose, scan_type='lookahead')
            
            if next_window is None:
                print(f"  No window visible - exploring...")
                
                currentPose, window_found = skills.explore(currentPose, came_from_left=came_from_left)
                
                if not window_found:
                    print(f"\n[ABORT] No more windows found")
                    return -1
            else:
                print(f"  [OK] Next window visible")
            
            print(f"\n[OK] Window {window_num + 1} complete!")
    
    print(f"\n[FORWARD JOURNEY COMPLETE]")
    return currentPose