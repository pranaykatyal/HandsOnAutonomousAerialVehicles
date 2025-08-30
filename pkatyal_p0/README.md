# RBE595 - Project 0 README.md HoAAV Orientation Estimation Project

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
- Navigate to: `/Phase1/Code/`
- Use:
  ```bash
  python3 Wrapper.py
  ```

## Output Locations

- **Rotplot Frames (3D orientation visualizations):**
  - Saved in: `Phase1/Code/Rotplot_frames/dataset_<dataset_num>/rotplot_overlapped.png`
  - Each dataset gets its own folder with the overlapped 3D orientation plot.

- **Orientation Comparison Plots (Yaw, Pitch, Roll vs Time):**
  - Saved in: `Phase1/Code/OrientationPlots/orientation_comparison_<dataset_num>.png`
  - Each dataset gets a separate comparison plot showing Vicon, Gyro, Accel, and Complementary Filter results.

- **VideoFrames (3D orientation videos for each method):**
  - Saved in: `Phase1/Code/VideoFrames/dataset_<dataset_num>/`
  - Each dataset folder contains videos for Gyro, Acc, CF, Vicon, and a combined 2x2 grid video if ffmpeg is installed.

## Notes
- All output folders are created automatically if they do not exist.
- You can change the number of frames or customize the plots by editing `Wrapper.py`.
- For questions or troubleshooting email me: pkatyal@wpi.edu
