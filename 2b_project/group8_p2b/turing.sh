#!/bin/bash

#SBATCH --mail-user=pkatyal@wpi.edu
#SBATCH --mail-type=ALL

#SBATCH -J p2b
#SBATCH --output=/home/pkatyal/logs/p2b_%j.out
#SBATCH --error=/home/pkatyal/logs/p2b_%j.err

#SBATCH -N 1
#SBATCH -n 32
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

python3 main.py maps/mapSplat.txt
