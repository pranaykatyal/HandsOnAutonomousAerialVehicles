#!/usr/bin/env python3
"""
Generate videos from logged frames after simulation completes
"""
import os
import sys
import glob
import shutil
import subprocess

def ffmpeging_video(input_pattern, output_file, fps=5):
    """Generate video from image sequence using ffmpeg"""
    # Ensure output file has .mp4 extension
    if not output_file.endswith('.mp4'):
        output_file += '.mp4'
    
    cmd = [
        'ffmpeg', '-y',
        '-pattern_type', 'glob',
        '-framerate', str(fps),
        '-i', input_pattern,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_file
    ]
    
    print(f"Running ffmpeg command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Successfully created video: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e}")
        print(f"ffmpeg stderr: {e.stderr}")
        return False

def create_combined_video(dataset_dir, video_paths):
    """
    Create a side-by-side video from two input videos
    """
    if len(video_paths) != 2:
        print("Error: Exactly 2 video paths required for side-by-side combination")
        return False
    
    output_path = os.path.join(dataset_dir, 'combined_visualization.mp4')
    
    cmd = [
        'ffmpeg', '-y',
        '-i', video_paths[0],
        '-i', video_paths[1],
        '-filter_complex', '[0:v][1:v]hstack=inputs=2[v]',
        '-map', '[v]',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error creating combined video: {e}")
        return False

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
    
    # Find all frames
    rgb_frames = sorted(glob.glob(os.path.join(log_dir, 'window_*_rgb.png')))
    seg_frames = sorted(glob.glob(os.path.join(log_dir, 'window_*_segmentation.png')))
    
    print(f"\nFound {len(rgb_frames)} RGB frames")
    print(f"Found {len(seg_frames)} segmentation frames")
    
    if not rgb_frames and not seg_frames:
        print("\nWarning: No frames found to create videos!")
        sys.exit(0)
    
    # Generate videos
    fps = 5  # Adjust frame rate as needed
    video_paths = []
    
    # RGB video
    if rgb_frames:
        print(f"\nGenerating RGB video at {fps} fps...")
        if ffmpeging_video(
            input_pattern=os.path.join(log_dir, 'window_*_rgb.png'),
            output_file=os.path.join(video_output_dir, 'rgb_video.mp4'),
            fps=fps
        ):
            video_paths.append(os.path.join(video_output_dir, 'rgb_video.mp4'))
    
    # Segmentation video
    if seg_frames:
        print(f"\nGenerating segmentation video at {fps} fps...")
        if ffmpeging_video(
            input_pattern=os.path.join(log_dir, 'window_*_segmentation.png'),
            output_file=os.path.join(video_output_dir, 'segmentation_video.mp4'),
            fps=fps
        ):
            video_paths.append(os.path.join(video_output_dir, 'segmentation_video.mp4'))
    
    # Create combined side-by-side video if both exist
    if len(video_paths) == 2:
        print(f"\nCreating combined side-by-side video...")
        # Scale both videos to 720p height while maintaining aspect ratio
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
            print(f"  - {vid}")

if __name__ == "__main__":
    main()