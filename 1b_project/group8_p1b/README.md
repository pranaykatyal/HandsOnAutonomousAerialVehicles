# RBE595 - Project 1b README.md HoAAV Unscented Kalman Filter

## Authors : Pranay Katyal, Anirudh Ramanathan, Hudson Kortus

## Requirements

To run `Wrapper.py` seamlessly, you need the following:

- Python 3.x
- Required Python packages:
  - numpy
  - scipy
  - matplotlib
- ffmpeg (for video creation)

### Install Python packages (if needed):
```bash
pip install numpy scipy matplotlib
```

### Install ffmpeg (Linux):
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

On other systems, see: https://ffmpeg.org/download.html

## How to Run Wrapper.py from the Command Line

1. **Run the main script:**
- Navigate to: `/group8_p1b/Code/`
- Use:
  ```bash
  python3 Wrapper.py
  ```
- It will  take about 5  minutes to complete for a smaller stride value, You can make the Stride size 100 for a faster ( shorter ) video production. 

## Output Locations

- **Rotplot Frames (3D orientation visualizations):**
  - Saved in: `Phase1/Code/Rotplot_frames/dataset_<dataset_num>/rotplot_overlapped.png`
  - Each dataset gets its own folder with the overlapped 3D orientation plot.

- **Orientation Comparison Plots (Yaw, Pitch, Roll vs Time):**
  - Saved in: `group8_p1b/Code/OrientationPlots/orientation_comparison_<dataset_num>.png`
  - Each dataset gets a separate comparison plot showing Vicon, Gyro, Accel, Complementary Filter, Madgwick Filter, and Unscented Kalman Filter results.

- **VideoFrames (3D orientation videos for each method):**
  - Saved in: `group8_p1b/Code/VideoFrames/dataset_<dataset_num>/`
  - Each dataset folder contains videos for Gyro, Acc, CF, Vicon, MF, UKF and a combined 6x1 grid video if ffmpeg is installed.

## Notes
- All output folders are created automatically if they do not exist.
- You can change the number of frames or customize the plots by editing `Wrapper.py`.
- For questions or troubleshooting email me: pkatyal@wpi.edu
