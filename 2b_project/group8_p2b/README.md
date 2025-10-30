# RBE595 - Project 2b - Navigation and 3D Scene Rendering

## Authors: Pranay Katyal, Anirudh Ramanathan, Hudson Kortus

## Project Overview
This project implements a complete quadrotor navigation system with:
- RRT* path planning in 3D environments
- B-spline trajectory generation
- Real-time simulation and visualization
- 3D scene rendering using Gaussian Splatting
- Apropriately uses VizFlyt and Turing clusters.

## Requirements

To run the project seamlessly, you need the following:

### Python Dependencies
```bash
conda create -n aerial_robotics python=3.10
conda activate aerial_robotics

# Core dependencies
pip install numpy scipy matplotlib pyquaternion

# Visualization and rendering
pip install open3d pytorch3d
```

### System Requirements
- CUDA-capable GPU (tested on V100, A100, H100)
- FFMPEG (for video generation)
- 64GB RAM recommended
- CUDA 12.4.0 or compatible version

## Running the Project

1. **Setup Environment:**
   ```bash
   module load cuda/12.4.0/3mdaov5
   module load miniconda3
   module load ffmpeg/6.1.1/cup2q2r
   conda activate aerial_robotics
   ```

2. **Run the Simulation:**
   - Navigate to the project directory:
     ```bash
     cd group8_p2b/
     ```
   - Run with default map:
     ```bash
     python3 main.py maps/mapSplat.txt
     ```
   - Or submit to SLURM: ( recommended ) (make sure the turing ID is updated in the turing.sh before running it or else it wont run)
     ```bash
     sbatch turing.sh
     ```

## Project Structure
- `main.py`: Entry point for simulation
- `simulator.py`: Core simulation and visualization
- `path_planner.py`: RRT* implementation
- `trajectory_generator.py`: B-spline trajectory generation
- `control.py`: Quadrotor controller
- `environment.py`: 3D environment and collision checking
- `splat_render.py`: Gaussian splatting renderer
- `maps/`: Environment definition files
- `renders/`: Output directory for rendered frames
- `report_outputs/`: Visualization outputs and plots
- `video_gen.py/` : Making Videos using images in Render Folder

## Output Locations

- **RRT* Path Planning:**
  - Final path visualization: `report_outputs/final_RRT_path.png`

- **Rendered Views:**
  - RGB frames: `renders/rgb/`
  - Depth maps: `renders/depth/`
  - Plot frames: `renders/plot/`

- **Log Files:**
  - Simulation logs: `log/`
  - SLURM outputs: `/home/[username]/logs/p2b_*.{out,err}`

## Features
- Adaptive RRT* parameters based on environment size
- Collision-free trajectory generation
- Real-time visualization of planning and execution
- Integration with Gaussian Splatting for photo-realistic rendering
- SLURM job submission support for HPC environments

## Notes
- Output directories are created automatically
- GPU memory usage scales with render resolution
- For questions or troubleshooting email: pkatyal@wpi.edu
