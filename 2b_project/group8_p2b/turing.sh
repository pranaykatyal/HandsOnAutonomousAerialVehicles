#!/bin/bash

#SBATCH --mail-user=hkortus@wpi.edu
#SBATCH --mail-type=ALL

#SBATCH -J p2b
#SBATCH --output=/home/hkortus/logs/p2b_%j.out
#SBATCH --error=/home/hkortus/logs/p2b_%j.err

#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -C H100|A100|V100|A30
#SBATCH -p academic
#SBATCH -t 2:00:00

module load cuda/12.4.0/3mdaov5
module load miniconda3

source "$("conda" info --base)/etc/profile.d/conda.sh"
conda activate /home/hkortus/.conda/envs/aerial_robotics

python3 /home/hkortus/RBE595/HandsOnAutonomousAerialVehicles/2b_project/group8_p2b/main.py maps/mapSplat.txt

