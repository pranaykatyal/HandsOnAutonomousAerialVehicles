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

def make_videos_for_dataset(dataset_dir, methods=['Gyro', 'Acc', 'CF', 'Vicon'], framerate=10):

    video_paths = []
    for method in methods:
        method_dir = os.path.join(dataset_dir, method)
        video_path = os.path.join(dataset_dir, f'{method}.mp4')
        # ffmpeg command to create video from frames
        cmd = [
            'ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(framerate),
            '-i', os.path.join(method_dir, 'frame_%04d.png'),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', video_path
        ]
        print(f'Creating video for {method} in {dataset_dir}...')
        subprocess.run(cmd, check=True)
        video_paths.append(video_path)
    # Combine 4 videos into a 2x2 grid (Gyro, Acc, CF, Vicon)
    if len(video_paths) == 4:
        combined_path = os.path.join(dataset_dir, 'combined.mp4')
        cmd_grid = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-i', video_paths[0], '-i', video_paths[1], '-i', video_paths[2], '-i', video_paths[3],
            '-filter_complex',
            '[0:v][1:v]hstack=inputs=2[top];[2:v][3:v]hstack=inputs=2[bottom];[top][bottom]vstack=inputs=2',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', combined_path
        ]
        print(f'Combining videos into grid for {dataset_dir}...')
        subprocess.run(cmd_grid, check=True)
        print(f'Combined video saved at {combined_path}')
    else:
        print('Not enough videos to combine into a grid.')


