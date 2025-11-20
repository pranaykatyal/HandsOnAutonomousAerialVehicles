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


def create_overlay_animation_video(overlay_dir='./run', frames_dir='./imgs', output_dir='./videos', fps=10):
    """
    Generate a video that alternates between overlay images (shown for 3 seconds)
    and their associated intermediate frames.
    
    Parameters:
    - overlay_dir: Directory containing overlay_j.png images
    - frames_dir: Directory containing _iter_j_frame_n.png images
    - output_dir: Directory to save output video
    - fps: Frames per second for video playback
    """
    import cv2
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all overlay images and sort them
    overlay_pattern = os.path.join(overlay_dir, 'overlay_*.png')
    overlay_files = sorted(glob.glob(overlay_pattern), key=natural_sort_key)
    
    if not overlay_files:
        print(f"No overlay images found in {overlay_dir}")
        return False
    
    print(f"\nFound {len(overlay_files)} overlay images")
    
    # Create temporary directory for sequenced frames
    temp_dir = '/tmp/overlay_animation_frames'
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    # Build sequence of frames
    frame_counter = 0
    
    for idx, overlay_file in enumerate(overlay_files):
        # Extract iteration number from overlay filename
        basename = os.path.basename(overlay_file)
        match = re.search(r'overlay_(\d+)', basename)
        if not match:
            continue
        iter_num = int(match.group(1))
        
        # Find all intermediate frames for this iteration
        frame_pattern = os.path.join(frames_dir, f'_iter_{iter_num:02d}_frame_*.png')
        iter_frames = sorted(glob.glob(frame_pattern), key=natural_sort_key)
        
        # Copy intermediate frames
        if iter_frames:
            print(f"Adding {len(iter_frames)} frames for iteration {iter_num}")
            for frame_file in iter_frames:
                # Read and write to ensure consistent format
                img = cv2.imread(frame_file)
                if img is not None:
                    # Ensure dimensions are even
                    h, w = img.shape[:2]
                    if h % 2 != 0 or w % 2 != 0:
                        new_h = h if h % 2 == 0 else h + 1
                        new_w = w if w % 2 == 0 else w + 1
                        img = cv2.resize(img, (new_w, new_h))
                    
                    output_frame = os.path.join(temp_dir, f'frame_{frame_counter:06d}.png')
                    cv2.imwrite(output_frame, img)
                    frame_counter += 1
        else:
            print(f"Warning: No frames found for iteration {iter_num}")
        
        # Add overlay image for 3 seconds (3 * fps frames)
        overlay_img = cv2.imread(overlay_file)
        if overlay_img is not None:
            # Resize overlay to match frame dimensions if needed
            if frame_counter > 0:
                # Get dimensions from first saved frame
                first_frame = cv2.imread(os.path.join(temp_dir, 'frame_000000.png'))
                if first_frame is not None and overlay_img.shape != first_frame.shape:
                    overlay_img = cv2.resize(overlay_img, 
                                            (first_frame.shape[1], first_frame.shape[0]))
            else:
                # Ensure dimensions are even for first overlay
                h, w = overlay_img.shape[:2]
                if h % 2 != 0 or w % 2 != 0:
                    new_h = h if h % 2 == 0 else h + 1
                    new_w = w if w % 2 == 0 else w + 1
                    overlay_img = cv2.resize(overlay_img, (new_w, new_h))
            
            overlay_duration_frames = 3 * fps
            print(f"Adding overlay_{iter_num}.png for {overlay_duration_frames} frames (3 seconds)")
            
            for _ in range(overlay_duration_frames):
                output_frame = os.path.join(temp_dir, f'frame_{frame_counter:06d}.png')
                cv2.imwrite(output_frame, overlay_img)
                frame_counter += 1
    
    # Add frames for the NEXT iteration after the last overlay
    if overlay_files:
        last_basename = os.path.basename(overlay_files[-1])
        last_match = re.search(r'overlay_(\d+)', last_basename)
        if last_match:
            last_iter_num = int(last_match.group(1))
            next_iter_num = last_iter_num + 1
            
            # Find frames for the next iteration
            next_frame_pattern = os.path.join(frames_dir, f'_iter_{next_iter_num:02d}_frame_*.png')
            next_iter_frames = sorted(glob.glob(next_frame_pattern), key=natural_sort_key)
            
            if next_iter_frames:
                print(f"Adding {len(next_iter_frames)} frames for final iteration {next_iter_num}")
                for frame_file in next_iter_frames:
                    img = cv2.imread(frame_file)
                    if img is not None:
                        # Ensure dimensions match
                        first_frame = cv2.imread(os.path.join(temp_dir, 'frame_000000.png'))
                        if first_frame is not None:
                            img = cv2.resize(img, (first_frame.shape[1], first_frame.shape[0]))
                        
                        output_frame = os.path.join(temp_dir, f'frame_{frame_counter:06d}.png')
                        cv2.imwrite(output_frame, img)
                        frame_counter += 1
    
    if frame_counter == 0:
        print("No frames to create video!")
        shutil.rmtree(temp_dir)
        return False
    
    print(f"\nTotal frames in video: {frame_counter}")
    print(f"Video duration: {frame_counter / fps:.1f} seconds at {fps} fps")
    
    # Use ffmpeg to create video with proper encoding
    output_path = os.path.join(output_dir, 'overlay_animation.mp4')
    
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', os.path.join(temp_dir, 'frame_%06d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'medium',
        '-crf', '23',
        output_path
    ]
    
    print(f"\nCreating video with ffmpeg...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Successfully created overlay animation video: {output_path}")
        
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e}")
        print(f"ffmpeg stderr: {e.stderr}")
        shutil.rmtree(temp_dir)
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