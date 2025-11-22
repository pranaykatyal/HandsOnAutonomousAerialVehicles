#!/usr/bin/env python3
"""
Combine RGB servo images with optical flow visualizations side-by-side
Creates separate videos for Phase 3 (alignment) and Phase 4 (approach), then combines them
"""

import cv2
import numpy as np
import glob
import os
import subprocess
import json


def combine_rgb_and_flow(log_dir='./log'):
    """
    Combine servo_iter_XX.png (RGB) and flow_iter_XX.png (flow) side-by-side
    """
    # Find all servo iteration images
    servo_files = sorted(glob.glob(os.path.join(log_dir, 'servo_iter_*.png')))
    
    if not servo_files:
        print("No servo images found!")
        return
    
    print(f"Found {len(servo_files)} servo images")
    
    # Extract iteration numbers
    servo_iterations = []
    for f in servo_files:
        basename = os.path.basename(f)
        iter_num = basename.replace('servo_iter_', '').replace('.png', '')
        servo_iterations.append(int(iter_num))
    servo_iterations.sort()
    print(f"Servo iterations: {servo_iterations}")
    
    combined_dir = os.path.join(log_dir, 'combined')
    os.makedirs(combined_dir, exist_ok=True)
    
    created_count = 0
    for servo_file in servo_files:
        # Extract iteration number
        basename = os.path.basename(servo_file)
        iter_num = basename.replace('servo_iter_', '').replace('.png', '')
        
        # Find corresponding flow image
        flow_file = os.path.join(log_dir, f'flow_iter_{iter_num}.png')
        
        if not os.path.exists(flow_file):
            print(f"Warning: No flow image for iteration {iter_num}")
            continue
        
        # Load images
        rgb_img = cv2.imread(servo_file)
        flow_img = cv2.imread(flow_file)
        
        if rgb_img is None or flow_img is None:
            print(f"Warning: Could not load images for iteration {iter_num}")
            continue
        
        # Resize flow to match RGB height
        h_rgb, w_rgb = rgb_img.shape[:2]
        h_flow, w_flow = flow_img.shape[:2]
        
        # Resize flow to match RGB height (maintain aspect ratio)
        scale = h_rgb / h_flow
        new_w_flow = int(w_flow * scale)
        flow_resized = cv2.resize(flow_img, (new_w_flow, h_rgb), interpolation=cv2.INTER_LINEAR)
        
        # Combine side-by-side
        combined = np.hstack([rgb_img, flow_resized])
        
        # Add text labels
        cv2.putText(combined, 'RGB + Detection', (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(combined, 'Optical Flow', (w_rgb + 20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        # Save combined image
        output_file = os.path.join(combined_dir, f'combined_iter_{iter_num}.png')
        cv2.imwrite(output_file, combined)
        created_count += 1
        
        if int(iter_num) % 5 == 0:
            print(f"  Created combined image for iteration {iter_num}")
    
    print(f"\n✓ Combined images saved to {combined_dir}/")
    print(f"✓ Successfully created {created_count} combined images")
    
    combined_files = sorted(glob.glob(os.path.join(combined_dir, 'combined_iter_*.png')))
    
    if not combined_files:
        print("No combined files created!")
        return
    
    # Get iteration numbers
    iteration_numbers = []
    for f in combined_files:
        basename = os.path.basename(f)
        iter_num = basename.replace('combined_iter_', '').replace('.png', '')
        iteration_numbers.append(int(iter_num))
    iteration_numbers.sort()
    
    print(f"Combined iterations: {iteration_numbers}")
    
    # Try to detect phase boundary by reading the images
    phase_boundary = detect_phase_boundary(combined_dir, iteration_numbers)
    
    if phase_boundary is None:
        print("Could not detect phase boundary automatically.")
        print(f"Please enter the iteration number where Phase 4 starts (first forward motion iteration):")
        try:
            phase_boundary = int(input("Phase 4 starts at iteration: "))
        except:
            print("Invalid input, using midpoint as boundary")
            phase_boundary = iteration_numbers[len(iteration_numbers) // 2]
    
    # Split iterations into phases
    phase3_iters = [i for i in iteration_numbers if i < phase_boundary]
    phase4_iters = [i for i in iteration_numbers if i >= phase_boundary]
    
    print(f"\nPhase 3 (Alignment) iterations: {phase3_iters} ({len(phase3_iters)} frames)")
    print(f"Phase 4 (Approach) iterations: {phase4_iters} ({len(phase4_iters)} frames)")
    
    # Create videos
    create_phase_videos(combined_dir, log_dir, phase3_iters, phase4_iters)


def detect_phase_boundary(combined_dir, iteration_numbers):
    """
    Detect where Phase 4 starts by looking for "Phase 4" text in images
    """
    print("\nDetecting phase boundary...")
    
    for i in iteration_numbers:
        img_path = os.path.join(combined_dir, f'combined_iter_{i:02d}.png')
        img = cv2.imread(img_path)
        
        if img is not None:
            # Extract top-left region where phase label appears
            # Phase 4 text is green (0, 255, 0) at around y=60, x=20
            roi = img[50:80, 15:150]  # Region where "Phase 4" text would be
            
            # Check for bright green pixels (Phase 4 marker)
            # Phase 4 text is drawn with color (0, 255, 0) in BGR
            green_mask = (roi[:, :, 1] > 200) & (roi[:, :, 0] < 100) & (roi[:, :, 2] < 100)
            green_ratio = green_mask.sum() / green_mask.size
            
            if green_ratio > 0.01:  # If more than 1% of ROI is bright green
                print(f"✓ Detected Phase 4 starting at iteration {i}")
                return i
    
    print("⚠ Could not auto-detect phase boundary")
    return None


def create_phase_videos(combined_dir, log_dir, phase3_iters, phase4_iters):
    """
    Create separate videos for Phase 3 and Phase 4, then concatenate
    """
    
    phase3_video = os.path.join(log_dir, 'Phase3_Alignment.mp4')
    phase4_video = os.path.join(log_dir, 'Phase4_Approach.mp4')
    combined_video = os.path.join(log_dir, 'CombinedVideo.mp4')
    
    videos_created = []
    
    # Create Phase 3 video
    if phase3_iters:
        print(f"\nCreating Phase 3 video ({len(phase3_iters)} frames)...")
        if create_video_from_iterations(combined_dir, phase3_iters, phase3_video):
            videos_created.append(phase3_video)
            print(f"✓ Phase 3 video: {phase3_video}")
    
    # Create Phase 4 video
    if phase4_iters:
        print(f"\nCreating Phase 4 video ({len(phase4_iters)} frames)...")
        if create_video_from_iterations(combined_dir, phase4_iters, phase4_video):
            videos_created.append(phase4_video)
            print(f"✓ Phase 4 video: {phase4_video}")
    
    # Concatenate videos
    if len(videos_created) == 2:
        print(f"\nConcatenating Phase 3 and Phase 4...")
        if concatenate_videos(videos_created, combined_video):
            print(f"✓ Combined video: {combined_video}")
            print(f"✓ Total frames: {len(phase3_iters) + len(phase4_iters)}")
            print(f"✓ Duration: {(len(phase3_iters) + len(phase4_iters)) / 5.0:.1f} seconds at 5 fps")
    elif len(videos_created) == 1:
        # Only one phase, rename it as combined
        os.rename(videos_created[0], combined_video)
        print(f"✓ Single-phase video saved as: {combined_video}")


def create_video_from_iterations(combined_dir, iterations, output_path):
    """
    Create video from specific iterations using FFmpeg
    """
    if not iterations:
        return False
    
    # Create file list
    file_list_path = output_path.replace('.mp4', '_files.txt')
    with open(file_list_path, 'w') as f:
        for i in iterations:
            img_path = os.path.join(combined_dir, f'combined_iter_{i:02d}.png')
            if os.path.exists(img_path):
                f.write(f"file '{os.path.abspath(img_path)}'\n")
                # IMPORTANT: Add duration for each frame (1/5 second = 0.2s for 5fps)
                f.write(f"duration 0.2\n")
        
        # Repeat last file without duration (FFmpeg concat requirement)
        if iterations:
            last_img = os.path.join(combined_dir, f'combined_iter_{iterations[-1]:02d}.png')
            if os.path.exists(last_img):
                f.write(f"file '{os.path.abspath(last_img)}'\n")
    
    # FFmpeg command - use concat demuxer with duration
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-f', 'concat', '-safe', '0',
        '-i', file_list_path,
        '-vsync', 'vfr',  # Variable framerate to respect durations
        '-pix_fmt', 'yuv420p',
        output_path
    ]
    
    try:
        subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
        # Don't remove file list for debugging
        # os.remove(file_list_path)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ FFmpeg error: {e.stderr}")
        # Fallback to OpenCV
        return create_video_opencv(combined_dir, iterations, output_path)
    except FileNotFoundError:
        print("✗ FFmpeg not found")
        return create_video_opencv(combined_dir, iterations, output_path)


def create_video_opencv(combined_dir, iterations, output_path):
    """
    Fallback: Create video using OpenCV
    """
    print(f"  Using OpenCV fallback...")
    
    if not iterations:
        return False
    
    # Read first image for dimensions
    first_img_path = os.path.join(combined_dir, f'combined_iter_{iterations[0]:02d}.png')
    first_img = cv2.imread(first_img_path)
    
    if first_img is None:
        print(f"✗ Could not read first image")
        return False
    
    h, w = first_img.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, 5.0, (w, h))
    
    for i in iterations:
        img_path = os.path.join(combined_dir, f'combined_iter_{i:02d}.png')
        img = cv2.imread(img_path)
        if img is not None:
            video.write(img)
    
    video.release()
    return True


def concatenate_videos(video_paths, output_path):
    """
    Concatenate multiple videos using FFmpeg
    """
    # Create concat file
    concat_file = output_path.replace('.mp4', '_concat.txt')
    with open(concat_file, 'w') as f:
        for video in video_paths:
            f.write(f"file '{os.path.abspath(video)}'\n")
    
    # FFmpeg concat
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
        '-i', concat_file,
        '-c', 'copy',  # Copy streams without re-encoding
        output_path
    ]
    
    try:
        subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
        os.remove(concat_file)  # Clean up
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Concatenation failed: {e.stderr}")
        # If copy codec fails, try re-encoding
        return concatenate_with_reencoding(video_paths, output_path)


def concatenate_with_reencoding(video_paths, output_path):
    """
    Fallback: Concatenate by re-encoding
    """
    print("  Trying concatenation with re-encoding...")
    
    concat_file = output_path.replace('.mp4', '_concat.txt')
    with open(concat_file, 'w') as f:
        for video in video_paths:
            f.write(f"file '{os.path.abspath(video)}'\n")
    
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
        '-i', concat_file,
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
        output_path
    ]
    
    try:
        subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
        os.remove(concat_file)
        return True
    except:
        return False


if __name__ == "__main__":
    combine_rgb_and_flow('./log')