"""
Drone Racing Navigation System - Phase-Based Architecture
Main orchestration file that conditionally runs forward and/or return journeys
"""

# =============================================================================
# CONFIGURATION FLAGS
# =============================================================================
GENERATE_VIDEO = False       # Set to False to skip video generation (saves time)
RUN_PHASE_1_2 = True        # Forward journey (windows 1→2→3→4)
RUN_PHASE_3 = True           # Return journey (windows 4→3→2→1)
# =============================================================================

from splat_render import SplatRenderer
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import json
import glob
import torch
import time

from windownavigator import WindowNavigator
from collisionChecker import doesItCollide


def main(renderer):
    """Main orchestration function"""
    os.makedirs('./log', exist_ok=True)
    os.makedirs('./log/frames', exist_ok=True)
    
    # Clean up old files
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
    print("DRONE RACING - PHASE-BASED ARCHITECTURE")
    print("="*70)
    print(f"Configuration:")
    print(f"  Phase 1-2 (Forward): {'ENABLED' if RUN_PHASE_1_2 else 'DISABLED'}")
    print(f"  Phase 3 (Return):    {'ENABLED' if RUN_PHASE_3 else 'DISABLED'}")
    print(f"  Video Generation:    {'ENABLED' if GENERATE_VIDEO else 'DISABLED'}")
    print("="*70 + "\n")

    # Reset collision checker
    import collisionChecker
    collisionChecker._default_checker = None
    
    # Initialize navigator
    navigator = WindowNavigator(renderer, device='cuda')
    
    # Initial pose
    currentPose = {
        'position': np.array([0.0, 0.0, 0.0]),
        'rpy': np.radians([0.0, 0.0, 0.0])
    }
    
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
    
    # Record initial frame
    rgb, _, _ = renderer.render(currentPose['position'], currentPose['rpy'])
    navigator.record_frame(rgb, pose=currentPose, annotation="START")
    
    # ==========================================================================
    # PHASE 1-2: FORWARD JOURNEY (Windows 1→4)
    # ==========================================================================
    start_time = time.time()

    if RUN_PHASE_1_2:
        print("\n" + "="*70)
        print("PHASE 1-2: FORWARD JOURNEY")
        print("="*70)
        
        from phase1_2_forward import run_forward_journey
        currentPose = run_forward_journey(navigator, renderer, currentPose)
        
        if currentPose == -1:
            print("\n[ABORT] Forward journey failed")
            return -1
        
        print(f"\n[OK] Forward journey complete")
        print(f"  Final position: {currentPose['position']}")
        print(f"  Final yaw: {np.degrees(currentPose['rpy'][2]):.1f}°")
    else:
        print("\n" + "="*70)
        print("PHASE 1-2: SKIPPED (simulating post-window-4 state)")
        print("="*70)
        
        # Simulate being past window 4 with 180° turn
        # Use [1.75, -0.02, 0.0] - offset left, Z at centerline
        currentPose = {
            'position': np.array([1.75, -0.02, 0.0]),  # Y=-0.02 (left), Z=0.0 (centerline)
            'rpy': np.array([0.0, 0.0, np.pi])         # 180° yaw
        }
        navigator.window_count = 4
        print(f"  Simulated position: {currentPose['position']}")
        print(f"  Simulated yaw: {np.degrees(currentPose['rpy'][2]):.1f}°")
    
    # ==========================================================================
    # PHASE 3: RETURN JOURNEY (Windows 4→1)
    # ==========================================================================
    if RUN_PHASE_3:
        print("\n" + "="*70)
        print("PHASE 3: RETURN JOURNEY")
        print("="*70)
        
        from phase3_return import run_return_journey
        currentPose = run_return_journey(navigator, renderer, currentPose)
        
        if currentPose == -1:
            print("\n[ABORT] Return journey failed")
            return -1
        
        print(f"\n[OK] Return journey complete")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Sim time taken to fly {elapsed_time} seconds")

    # ==========================================================================
    # SAVE RESULTS
    # ==========================================================================
    
    with open('./log/pose_history.json', 'w') as f:
        json.dump(navigator.pose_history, f, indent=2, default=str)
    
    navigator.save_frames_summary()
    
    # Video generation
    if GENERATE_VIDEO and len(navigator.video_frames) > 0:
        print("\n" + "="*70)
        print("GENERATING VIDEO")
        print("="*70)
        
        try:
            import subprocess
            
            print(f"  Creating video from {len(navigator.video_frames)} frames...")
            
            cmd = [
                'ffmpeg', '-y',
                '-framerate', '10',
                '-i', './log/frames/frame_%04d.png',
                '-c:v', 'mpeg4',
                '-pix_fmt', 'yuv420p',
                '-qscale:v', '3',
                './log/navigation_video.mp4'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"  [OK] Video saved: ./log/navigation_video.mp4")
            else:
                print(f"  [WARN] ffmpeg failed: {result.stderr}")
        except FileNotFoundError:
            print(f"  [WARN] ffmpeg not found, skipping video generation")
        except Exception as e:
            print(f"  [WARN] Video generation failed: {e}")
    
    print("\n" + "="*70)
    print("NAVIGATION COMPLETE")
    print(f"  Windows passed: {navigator.window_count}")
    print("="*70 + "\n")


if __name__ == "__main__":
    config_path = "../data/P5_colmap_splat/P5_colmap/splatfacto/2025-11-17_130359/config.yml"
    json_path = "../data/render_settings/render_settings.json"

    renderer = SplatRenderer(config_path, json_path)
    main(renderer)