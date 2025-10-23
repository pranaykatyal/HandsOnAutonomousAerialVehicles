
# HandsOnAutonomousAerialVehicles
# Group 8
## How to add and push a specific folder to this repository

If you want to add and push only a particular folder (e.g., `pkatyal_p0`) to the repository, use the following commands:

```bash
cd ~/HoAAV
git add #Folder name
git commit -m "Add pkatyal_p0 folder"
git push origin main
```

This will stage, commit, and push only the changes in the `pkatyal_p0` folder to the remote repository.
## How to activate the HoAAV Python virtual environment



To activate the environment:
```bash
source ~/HoAAV/HoAAV/bin/activate
```


# How to connect to Turing and activate your Conda environment

To connect to Turing and activate your Conda environment for this project:

```bash
ssh pkatyal@turing.wpi.edu
module load miniconda3/25.1.1/24g7bpu  # or the latest available miniconda3 module
conda activate aerial_robotics
```


# Using GPUs and Job Management on Turing

## Loading CUDA

To load CUDA (e.g., version 12.4.0):
```bash
module load cuda/12.4.0/3mdaov5
```

## Starting an Interactive GPU Session

To request an interactive session with specific resources:
```bash
sinteractive
```
You will be prompted for resource requirements. Example:
```
How many CPU cores? :: 4
How many GPUs? :: 2
GPU Type (or leave blank for any)? ::
How much memory (in MB)? :: 8192
How many minutes do you need? :: 120
What partition? :: academic
```

## Checking GPU and CUDA Status

After your session starts, you can check GPU status and CUDA version:
```bash
nvidia-smi         # Check GPU usage and status
nvcc --version     # Check CUDA version
```

## Monitoring and Managing Jobs

To see your currently queued/running jobs:
```bash
squeue --me
```

To cancel a job:
```bash
scancel [jobid]
```
Replace `[jobid]` with the job ID shown in `squeue`.


# Copying Files Between Local and Turing

## To copy the `2b_project` folder from your local machine to Turing

Run this command **in your local terminal** (not in an SSH session):
```bash
scp -r ~/HoAAV/2b_project pkatyal@turing.wpi.edu:~/
```

## To copy the `2b_project` folder from Turing back to your local machine

Run this command **in your local terminal** (not in an SSH session):
```bash
scp -r pkatyal@turing.wpi.edu:~/2b_project ~/HoAAV/
```

These commands use `scp` (secure copy) to transfer files between your local computer and Turing. Always run them from your local terminal.


