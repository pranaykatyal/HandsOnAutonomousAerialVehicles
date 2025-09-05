import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Polygon
import subprocess
import os


def rotplot(R, currentAxes=None):
    # This is a simple function to plot the orientation
    # of a 3x3 rotation matrix R in 3-D
    # You should modify it as you wish for the project.

    lx = 3.0
    ly = 1.5
    lz = 1.0

    x = .5 * np.array([[+lx, -lx, +lx, -lx, +lx, -lx, +lx, -lx],
                       [+ly, +ly, -ly, -ly, +ly, +ly, -ly, -ly],
                       [+lz, +lz, +lz, +lz, -lz, -lz, -lz, -lz]])

    xp = np.dot(R, x)
    ifront = np.array([0, 2, 6, 4, 0])
    iback = np.array([1, 3, 7, 5, 1])
    itop = np.array([0, 1, 3, 2, 0])
    ibottom = np.array([4, 5, 7, 6, 4])

    if currentAxes:
        ax = currentAxes
    else:
        fig = plt.figure()
        # ax = fig.gca(projection='3d')
        ax = fig.add_subplot(111, projection='3d')

    ax.plot(xp[0, itop], xp[1, itop], xp[2, itop], 'k-')
    ax.plot(xp[0, ibottom], xp[1, ibottom], xp[2, ibottom], 'k-')

    rectangleFront = a3.art3d.Poly3DCollection([list(zip(xp[0, ifront], xp[1, ifront], xp[2, ifront]))])
    rectangleFront.set_facecolor('r')
    ax.add_collection(rectangleFront)

    rectangleBack = a3.art3d.Poly3DCollection([list(zip(xp[0, iback], xp[1, iback], xp[2, iback]))])
    rectangleBack.set_facecolor('b')
    ax.add_collection(rectangleBack)

    ax.set_aspect('equal')
    ax.set_xlim3d(-2, 2)
    ax.set_ylim3d(-2, 2)
    ax.set_zlim3d(-2, 2)

    return ax


# Example usage: Putting two rotations on one graph.
# Call the function below from another Python file.

# from rotplot import rotplot
REye = np.eye(3)
myAxis = rotplot(REye)
RTurn = np.array([[np.cos(np.pi / 2), 0, np.sin(np.pi / 2)], [0, 1, 0], [-np.sin(np.pi / 2), 0, np.cos(np.pi / 2)]])
rotplot(RTurn, myAxis)
plt.show()


def make_videos_for_dataset(dataset_dir, methods=['Gyro', 'Acc', 'CF', 'Vicon', 'Madgwick'], framerate=10):
    """
    Create individual videos for each method and combine them into a grid layout.
    
    Args:
        dataset_dir: Path to the dataset directory containing method folders
        methods: List of method names to process
        framerate: Framerate for the output videos
    """
    video_paths = []
    successful_methods = []
    
    for method in methods:
        method_dir = os.path.join(dataset_dir, method)
        
        # Check if method directory exists and has frames
        if not os.path.exists(method_dir):
            print(f'Method directory {method_dir} does not exist, skipping {method}...')
            continue
            
        # Check if there are any frame files
        frame_files = [f for f in os.listdir(method_dir) if f.startswith('frame_') and f.endswith('.png')]
        if not frame_files:
            print(f'No frame files found in {method_dir}, skipping {method}...')
            continue
            
        video_path = os.path.join(dataset_dir, f'{method}.mp4')
        
        # ffmpeg command to create video from frames
        cmd = [
            'ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(framerate),
            '-i', os.path.join(method_dir, 'frame_%04d.png'),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', video_path
        ]
        
        print(f'Creating video for {method} in {dataset_dir}...')
        
        try:
            subprocess.run(cmd, check=True)
            video_paths.append(video_path)
            successful_methods.append(method)
            print(f'Successfully created {method}.mp4')
        except subprocess.CalledProcessError as e:
            print(f'Error creating video for {method}: {e}')
        except FileNotFoundError:
            print(f'ffmpeg not found. Please install ffmpeg to create videos.')
            return
    
    # Create combined grid video based on number of successful videos
    if len(video_paths) >= 4:
        create_combined_video(dataset_dir, video_paths, successful_methods)
    else:
        print(f'Created {len(video_paths)} individual videos. Need at least 4 videos to create combined grid.')


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