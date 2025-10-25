#!/bin/bash

#SBATCH --mail-user=pkatyal@wpi.edu
#SBATCH --mail-type=ALL

#SBATCH -J p2b
#SBATCH --output=/home/pkatyal/logs/p2b_%j.out
#SBATCH --error=/home/pkatyal/logs/p2b_%j.err

#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -C H100|A100|V100|A30
#SBATCH -p academic
#SBATCH -t 2:00:00

module load cuda/12.4.0/3mdaov5
module load ffmpeg/6.1.1/cup2q2r

module load miniconda3

source "$("conda" info --base)/etc/profile.d/conda.sh"
conda activate /home/pkatyal/.conda/envs/aerial_robotics

python3 /home/pkatyal/HandsOnAutonomousAerialVehicles/2b_project/group8_p2b/main.py maps/mapSplat2.txt

# python3 -c "import video_gen; video_gen.ffmpeging_video(
# '/home/hkortus/RBE595/HandsOnAutonomousAerialVehicles/2b_project/group8_p2b/renders',
# 'depth_', 
# 10, 
# '/home/hkortus/RBE595/HandsOnAutonomousAerialVehicles/2b_project/group8_p2b/report_outputs'
# )"

# python3 -c "import video_gen; video_gen.ffmpeging_video(
# '/home/hkortus/RBE595/HandsOnAutonomousAerialVehicles/2b_project/group8_p2b/renders',
# 'plot_', 
# 10, 
# '/home/hkortus/RBE595/HandsOnAutonomousAerialVehicles/2b_project/group8_p2b/report_outputs'
# )"

# python3 -c "import video_gen; video_gen.create_combined_video(
#     '/home/hkortus/RBE595/HandsOnAutonomousAerialVehicles/2b_project/group8_p2b/report_outputs',
#     [
#         '/home/hkortus/RBE595/HandsOnAutonomousAerialVehicles/2b_project/group8_p2b/report_outputs/rgb_.mp4', 
#         '/home/hkortus/RBE595/HandsOnAutonomousAerialVehicles/2b_project/group8_p2b/report_outputs/plot_.mp4'
#     ]
#     )"
