#!/usr/bin/env python3
"""
Combines normal, flow, and detection frames into a single horizontal video.
Each type of frame persists until a new one is available.
"""

import os
import re
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Configuration
FRAMES_DIR = Path("/home/hkortus/RBE595/HandsOnAutonomousAerialVehicles/5_project/group8_p5/Code/log/frames")
OUTPUT_VIDEO = FRAMES_DIR.parent / "combined_video2.mp4"
FPS = 10  # Adjust as needed
MAX_HEIGHT = 720  # Resize frames to this height for more manageable video size

def extract_frame_number(filename):
    """Extract frame number from filename."""
    match = re.search(r'_(\d+)\.png$', filename)
    if match:
        return int(match.group(1))
    match = re.search(r'^frame_(\d+)\.png$', filename)
    if match:
        return int(match.group(1))
    return None

def get_frame_files(frames_dir):
    """Get all frame files organized by type and index."""
    frames = {'normal': {}, 'flow': {}, 'detection': {}}
    
    for filename in os.listdir(frames_dir):
        if not filename.endswith('.png'):
            continue
        
        frame_num = extract_frame_number(filename)
        if frame_num is None:
            continue
        
        filepath = frames_dir / filename
        
        if filename.startswith('flow_frame_'):
            frames['flow'][frame_num] = filepath
        elif filename.startswith('detection_frame_'):
            frames['detection'][frame_num] = filepath
        elif filename.startswith('frame_'):
            frames['normal'][frame_num] = filepath
    
    return frames

def resize_to_match_height(images, target_height):
    """Resize all images to have the same height while maintaining aspect ratio."""
    resized = []
    for img in images:
        if img is None:
            return None
        h, w = img.shape[:2]
        scale = target_height / h
        new_w = int(w * scale)
        resized_img = cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_LINEAR)
        resized.append(resized_img)
    return resized

def create_combined_video(frames_dir, output_path, fps=30):
    """Create a video combining normal, flow, and detection frames horizontally."""
    
    print("Scanning frames directory...")
    frames = get_frame_files(frames_dir)
    
    # Find the range of frame indices
    all_indices = set()
    for frame_type in frames.values():
        all_indices.update(frame_type.keys())
    
    if not all_indices:
        print("No frames found!")
        return
    
    # Get all normal frame indices (these drive the video)
    normal_indices = sorted(frames['normal'].keys())
    if not normal_indices:
        print("No normal frames found!")
        return
    
    # Get sorted flow and detection indices for efficient lookup
    flow_indices = sorted(frames['flow'].keys())
    detection_indices = sorted(frames['detection'].keys())
    
    min_idx = min(normal_indices)
    max_idx = max(normal_indices)
    
    print(f"Frame range: {min_idx} to {max_idx}")
    print(f"Normal frames: {len(frames['normal'])}")
    print(f"Flow frames: {len(frames['flow'])}")
    print(f"Detection frames: {len(frames['detection'])}")
    
    # Track the last available frame for each type
    last_frames = {
        'normal': None,
        'flow': None,
        'detection': None
    }
    
    # Track current index for flow and detection
    current_flow_idx = 0
    current_detection_idx = 0
    
    # Create temporary directory for combined frames
    temp_dir = frames_dir / "temp_combined"
    temp_dir.mkdir(exist_ok=True)
    
    print("\nCreating combined frames...")
    frames_written = 0
    
    # Iterate through all normal frames (not skipping any)
    for idx in tqdm(normal_indices):
        # Always load the normal frame
        if idx in frames['normal']:
            img = cv2.imread(str(frames['normal'][idx]))
            if img is not None:
                last_frames['normal'] = img
        
        # Find the most recent flow frame up to current index
        while current_flow_idx < len(flow_indices) and flow_indices[current_flow_idx] <= idx:
            flow_idx = flow_indices[current_flow_idx]
            img = cv2.imread(str(frames['flow'][flow_idx]))
            if img is not None:
                last_frames['flow'] = img
            current_flow_idx += 1
        
        # Find the most recent detection frame up to current index
        while current_detection_idx < len(detection_indices) and detection_indices[current_detection_idx] <= idx:
            detection_idx = detection_indices[current_detection_idx]
            img = cv2.imread(str(frames['detection'][detection_idx]))
            if img is not None:
                last_frames['detection'] = img
            current_detection_idx += 1
        
        # Create placeholder for flow if not yet available
        if last_frames['flow'] is None and last_frames['normal'] is not None:
            h, w = last_frames['normal'].shape[:2]
            last_frames['flow'] = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(last_frames['flow'], 'Waiting for flow...', 
                       (w//4, h//2), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
        
        # Create placeholder for detection if not yet available
        if last_frames['detection'] is None and last_frames['normal'] is not None:
            h, w = last_frames['normal'].shape[:2]
            last_frames['detection'] = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(last_frames['detection'], 'Waiting for detection...', 
                       (w//4, h//2), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
        
        # Skip if we somehow don't have a normal frame
        if last_frames['normal'] is None:
            continue
        
        # Get current frames (using last available)
        current_frames = [
            last_frames['normal'],
            last_frames['flow'],
            last_frames['detection']
        ]
        
        # Resize all frames to match height
        heights = [img.shape[0] for img in current_frames]
        target_height = min(heights)  # Use minimum height to avoid upscaling
        target_height = min(target_height, MAX_HEIGHT)  # Cap at MAX_HEIGHT
        resized_frames = resize_to_match_height(current_frames, target_height)
        
        if resized_frames is None:
            continue
        
        # Concatenate horizontally
        combined_frame = np.hstack(resized_frames)
        
        # Save combined frame to temp directory
        frame_path = temp_dir / f"combined_{frames_written:06d}.png"
        cv2.imwrite(str(frame_path), combined_frame)
        frames_written += 1
    
    print(f"\nTotal frames created: {frames_written}")
    
    # Use ffmpeg to create video from frames
    print("\nCreating video with ffmpeg...")
    import subprocess
    try:
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', str(temp_dir / 'combined_%06d.png'),
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',  # Enable fast start for streaming
            str(output_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Video saved to: {output_path}")
            # Clean up temporary frames
            import shutil
            shutil.rmtree(temp_dir)
            print("Temporary frames cleaned up.")
        else:
            print("FFmpeg failed to create video.")
            print(result.stderr)
    except FileNotFoundError:
        print("FFmpeg not found. Please install it: sudo apt-get install ffmpeg")
        print(f"Combined frames saved to: {temp_dir}")
        print("You can manually create the video using ffmpeg with the command above.")

if __name__ == "__main__":
    create_combined_video(FRAMES_DIR, OUTPUT_VIDEO, FPS)
    print("\nDone!")
