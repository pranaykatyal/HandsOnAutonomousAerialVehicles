# Installing VizFlyt 

## Turing setup
* refer to - https://docs.turing.wpi.edu/
* setting CUDA12.4 - **module load cuda/12.4.0/3mdaov5**

## Conda installation (create your conda environment on Turing preferably)
* conda create -n aerial_robotics python=3.10
* conda activate aerial_robotics

## Installing dependencies
* pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 
* Please watch **Turing** tutorial video for setting up CUDA version on Turing. Make sure you install CUDA 12.4 
* Load CUDA on turing -  module load cuda/12.4.0/3mdaov5

## Steps to install VizFlyt without ROS support
* git clone https://github.com/pearwpi/VizFlyt.git
* cd VizFlyt/nerfstudio
* pip install --upgrade pip setuptools
* pip install -e .

