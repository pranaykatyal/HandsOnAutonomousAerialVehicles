"""
Phase 3: Return Journey
Navigate back through windows 4→3→2→1
"""

import numpy as np
from returnskills import ReturnNavigationSkills


def run_return_journey(navigator, renderer, currentPose):
    """
    Execute return journey through windows 4→3→2→1
    
    Args:
        navigator: WindowNavigator instance
        renderer: SplatRenderer instance
        currentPose: Current drone pose dict (should be facing 180° after turnback)
        
    Returns:
        currentPose: Updated pose after completing return journey, or -1 on failure
    """
    print("\n[RETURN JOURNEY] Starting navigation through windows 4→1")
    print(f"Current position: {currentPose['position']}")
    print(f"Current yaw: {np.degrees(currentPose['rpy'][2]):.1f}°")
    
    # Initialize return skills
    return_skills = ReturnNavigationSkills(navigator)
    
    # Return through 4 windows in reverse order
    return_window_order = [4, 3, 2, 1]
    
    for return_win_num in return_window_order:
        print(f"\n{'='*70}")
        print(f"RETURN WINDOW {return_win_num} / 4")
        print(f"{'='*70}")
        
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
            
            # FIX_YAW
            result = return_skills.fix_yaw_return(currentPose, window_3d_pos)
            if result == -1:
                print(f"\n[ABORT] Return yaw alignment failed")
                return -1
            currentPose = result
            
            # ALIGN
            currentPose, error_mag = return_skills.align_return(
                currentPose, window_3d_pos, corners_2d
            )
            
            print(f"  Alignment error: {error_mag:.1f}px")
            
            # APPROACH
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
    print(f"  Final yaw: {np.degrees(currentPose['rpy'][2]):.1f}°")
    print(f"{'='*70}")
    
    return currentPose