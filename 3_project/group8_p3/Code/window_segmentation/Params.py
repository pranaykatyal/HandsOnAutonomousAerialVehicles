import os

HOME_PATH   =   os.path.expanduser("~")
JOB_ID      =   "window_seg_big"
MODEL_NAME  =   "windowseg"
DS_PATH     =   "/home/hkortus/scratch/window_images/"
OUT_PATH    =   "/home/hkortus/RBE595/HandsOnAutonomousAerialVehicles/3_project/group8_p3/Code/window_segmentation/logs/"

JOB_FOLDER  =   os.path.join(OUT_PATH, JOB_ID)
TRAINED_MDL_PATH    =   os.path.join(JOB_FOLDER, "parameters")
BATCH_SIZE          =   32
LR                  =   1e-6
LOG_BATCH_INTERVAL  =   1
LOG_WANDB = True
NUM_WORKERS  =   32