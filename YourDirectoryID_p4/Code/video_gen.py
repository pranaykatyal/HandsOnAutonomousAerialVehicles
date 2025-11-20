#!/usr/bin/env python3
"""
Generate videos from logged frames after simulation completes
"""
import os
import sys
import glob
import shutil
import subprocess
import re

def natural_sort_key(s):
    """Sort strings containing numbers naturally"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def ffmpeging_video(input_files, output_file, fps=5):
    """Generate video from sorted image list using ffmpeg"""
    if not output_file.endswith('.mp4'):
        output_file += '.mp4'
    
    # Create a temporary file list for ffmpeg
    file_list_path = '/tmp/ffmpeg_filelist.txt'
    with open(file_list_path, 'w') as f:
        for img_file in input_files:
            # ffmpeg needs file paths to be relative or absolute
            f.write(f"file '{os.path.abspath(img_file)}'\n")
    
    cmd = [
        'ffmpeg', '-y',
        '-f', 'concat',
        '-safe', '0',
        '-r', str(fps),
        '-i', file_list_path,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_file
    ]
    
    print(f"Creating video from {len(input_files)} frames at {fps} fps...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Successfully created video: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e}")
        print(f"ffmpeg stderr: {e.stderr}")
        return False
    finally:
        if os.path.exists(file_list_path):
            os.remove(file_list_path)

def main():
    # Directories
    log_dir = './log'
    video_output_dir = './videos'
    
    # Clear and recreate videos directory
    if os.path.exists(video_output_dir):
        shutil.rmtree(video_output_dir)
    os.makedirs(video_output_dir)
    
    print("=" * 60)
    print("Starting video generation from logged frames")
    print("=" * 60)
    
    # Check if log directory exists
    if not os.path.exists(log_dir):
        print(f"Error: Log directory '{log_dir}' does not exist!")
        sys.exit(1)
    
    # Find all frames - including both alignment and intermediate frames
    # Pattern matches: window_X_iter_Y_align_rgb.png and window_X_iter_Y_frame_Z_rgb.png
    # Exclude frames with window_-1 in their name
    all_rgb_frames = glob.glob(os.path.join(log_dir, '*_rgb.png'))
    rgb_frames = sorted([
        f for f in all_rgb_frames
        if not (
            'window_-1' in f or 'window_-1_' in f or 'nav_frame' in f
        )
    ], key=natural_sort_key)
    all_seg_frames = glob.glob(os.path.join(log_dir, '*_segmentation.png'))
    seg_frames = sorted([
        f for f in all_seg_frames
        if not (
            'window_-1' in f or 'window_-1_' in f or 'nav_frame' in f
        )
    ], key=natural_sort_key)
    
    print(f"\nFound {len(rgb_frames)} RGB frames")
    print(f"Found {len(seg_frames)} segmentation frames")
    
    if not rgb_frames and not seg_frames:
        print("\nWarning: No frames found to create videos!")
        sys.exit(0)
    
    # Show breakdown of frame types
    align_frames = [f for f in rgb_frames if '_align_' in f]
    intermediate_frames = [f for f in rgb_frames if '_frame_' in f]
    print(f"  - {len(align_frames)} alignment frames")
    print(f"  - {len(intermediate_frames)} intermediate motion frames")
    
    # Generate videos
    fps = 10  # Adjust frame rate as needed
    video_paths = []
    
    # RGB video
    if rgb_frames:
        print(f"\nGenerating RGB video at {fps} fps...")
        if ffmpeging_video(
            input_files=rgb_frames,
            output_file=os.path.join(video_output_dir, 'rgb_video.mp4'),
            fps=fps
        ):
            video_paths.append(os.path.join(video_output_dir, 'rgb_video.mp4'))
    
    # Segmentation video
    if seg_frames:
        print(f"\nGenerating segmentation video at {fps} fps...")
        if ffmpeging_video(
            input_files=seg_frames,
            output_file=os.path.join(video_output_dir, 'segmentation_video.mp4'),
            fps=fps
        ):
            video_paths.append(os.path.join(video_output_dir, 'segmentation_video.mp4'))
    
    # Create combined side-by-side video if both exist
    if len(video_paths) == 2:
        print(f"\nCreating combined side-by-side video...")
        cmd = [
            'ffmpeg', '-y',
            '-i', video_paths[0],
            '-i', video_paths[1],
            '-filter_complex',
            '[0:v]scale=-1:720[v0];[1:v]scale=-1:720[v1];[v0][v1]hstack=inputs=2[v]',
            '-map', '[v]',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            os.path.join(video_output_dir, 'combined_visualization.mp4')
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("Successfully created combined video")
        except subprocess.CalledProcessError as e:
            print(f"Error creating combined video: {e}")
            print(f"ffmpeg stderr: {e.stderr}")
    
    print("\n" + "=" * 60)
    print("Video generation complete!")
    print(f"Videos saved in: {video_output_dir}/")
    print("=" * 60)
    
    # List generated videos
    videos = [f for f in os.listdir(video_output_dir) if f.endswith('.mp4')]
    if videos:
        print("\nGenerated videos:")
        for vid in videos:
            size_mb = os.path.getsize(os.path.join(video_output_dir, vid)) / (1024 * 1024)
            print(f"  - {vid} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()