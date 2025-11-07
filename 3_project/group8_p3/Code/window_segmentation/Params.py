import os

HOME_PATH   =   os.path.expanduser("~")
JOB_ID      =   "window_seg_2_new_datalaoder"
MODEL_NAME  =   "windowseg"
DS_PATH     =   "/home/hkortus/scratch/window_images/"
OUT_PATH    =   "/home/hkortus/RBE595/HandsOnAutonomousAerialVehicles/3_project/group8_p3/Code/window_segmentation/logs/"
PRETRAINED_PATH = "/home/hkortus/RBE595/HandsOnAutonomousAerialVehicles/3_project/group8_p3/Code/window_segmentation/logs/window_seg_small_dataloader_2_1/parameter3/19.pth"

JOB_FOLDER  =   os.path.join(OUT_PATH, JOB_ID)
BATCH_SIZE          =   32
LR                  =   4e-6
LOG_BATCH_INTERVAL  =   1
LOG_WANDB = True
NUM_WORKERS  =   20

