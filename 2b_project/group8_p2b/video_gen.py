import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Polygon
import subprocess
import os

def gen_vizflyt(img_dir,fps,save_dir):
    # Check if directory exists and has frames
    if not os.path.exists(img_dir):
        print(f'directory {img_dir} does not exist!')
        return False
    
    if not os.path.exists(save_dir):
        print(f'directory {save_dir} did not exist so i created it myself')
        return False

    # Check if there are any frame files
    depth_files = [f for f in os.listdir(img_dir) if f.startswith('depth_') and f.endswith('.png')]
    rgb_files = [f for f in os.listdir(img_dir) if f.startswith('rgb_') and f.endswith('.png')]
    if not depth_files:
        print(f'No depth_files')
        return False
    if not rgb_files:
        print(f'No depth_files')
        return False
        
    video_path = os.path.join(save_dir, 'VizFlyt_FPV.mp4')
    
    # ffmpeg command to create video from frames
    cmd = [
        'ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps),
        '-pattern_type', 'glob', '-i', os.path.join(img_dir, 'rgb_*_*.png'),
        '-vf', 'hflip,vflip',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', video_path
    ]
    
    print(f'Creating video with cmd {cmd}...')
    
    try:
        subprocess.run(cmd, check=True)
        print(f'Successfully created {video_path}')
    except subprocess.CalledProcessError as e:
        print(f'Error creating video for {video_path}: {e}')
    except FileNotFoundError:
        print(f'ffmpeg not found. Please install ffmpeg to create videos.')
        return
    



def create_combined_video(dataset_dir, video_paths, method_names):
    """
    Create a combined grid video from individual method videos.
    
    Args:
        dataset_dir: Directory to save the combined video
        video_paths: List of paths to individual videos
        method_names: List of method names corresponding to video paths
    """
    combined_path = os.path.join(dataset_dir, 'combined.mp4')
    
    if len(video_paths) == 4:
        # 2x2 grid for 4 videos
        cmd_grid = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-i', video_paths[0], '-i', video_paths[1], '-i', video_paths[2], '-i', video_paths[3],
            '-filter_complex',
            '[0:v][1:v]hstack=inputs=2[top];[2:v][3:v]hstack=inputs=2[bottom];[top][bottom]vstack=inputs=2',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', combined_path
        ]
        layout = "2x2"
        
    elif len(video_paths) == 5:
        # 5 videos in a horizontal row - simple and clean
        cmd_grid = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-i', video_paths[0], '-i', video_paths[1], '-i', video_paths[2], 
            '-i', video_paths[3], '-i', video_paths[4],
            '-filter_complex',
            '[0:v][1:v][2:v][3:v][4:v]hstack=inputs=5',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', combined_path
        ]
        layout = "1x5 horizontal"
        
    elif len(video_paths) == 6:
        # 3x2 grid for 6 videos
        cmd_grid = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-i', video_paths[0], '-i', video_paths[1], '-i', video_paths[2], 
            '-i', video_paths[3], '-i', video_paths[4], '-i', video_paths[5],
            '-filter_complex',
            '[0:v][1:v][2:v]hstack=inputs=3[top];[3:v][4:v][5:v]hstack=inputs=3[bottom];[top][bottom]vstack=inputs=2',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', combined_path
        ]
        layout = "3x2"
        
    else:
        # For other numbers, create a horizontal stack
        inputs = ''.join([f'-i {path} ' for path in video_paths])
        filter_str = ''.join([f'[{i}:v]' for i in range(len(video_paths))]) + f'hstack=inputs={len(video_paths)}'
        
        cmd_grid = [
            'ffmpeg', '-y', '-loglevel', 'error'
        ] + [item for path in video_paths for item in ['-i', path]] + [
            '-filter_complex', filter_str,
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', combined_path
        ]
        layout = f"1x{len(video_paths)}"
    
    print(f'Combining videos into {layout} grid for {dataset_dir}...')
    print(f'Methods in grid: {", ".join(method_names)}')
    
    try:
        subprocess.run(cmd_grid, check=True)
        print(f'Combined video saved at {combined_path}')
    except subprocess.CalledProcessError as e:
        print(f'Error creating combined video: {e}')
    except FileNotFoundError:
        print(f'ffmpeg not found. Please install ffmpeg to create combined video.')