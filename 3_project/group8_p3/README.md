# Project3 -- Group8  -- Hudson Kortus, Anirudh Ramanathan, Pranay Katyal


## Structure
```text
├── Code
│   ├── main.py                 # Main entry point 
│   ├── splat_render.py         # Gaussian splat renderer: generates RGB + depth from pose
│   ├── control.py              # PID controller and QuadrotorController class
│   ├── quad_dynamics.py        # Quadrotor dynamics simulation
│   ├── tello.py                # DJI Tello drone parameters
|   ├── log/                    # Controller logs (auto-created) + Renderer Images are saved here.
│   └── trajectory_generator.py # B-spline trajectory generation
├── data
│   ├── washburn-env6-itr0-1fps/           # Splat rendering checkpoint
│   └── washburn-env6-itr0-1fps_nf_format/ # COLMAP image dataset
├── render_settings
│   └── render_settings_2.json  # Render configuration (resolution, FOV, camera matrix)
├── Video.mp4
├── README.md
└── Report.pdf


```
## How to run?
- Navigate to `Code/`folder in a Turing cluster, and write `sbatch turing.sh` . Note : you may need to modify the path according to your turing directory setup. 
- You will  also need to Modify `params.py` to update the path of the UNet model `1.pth`. 
- Use `squeue --me` to check updates on the model.
- Once you run the SLURM script, it should take about `5 minutes` to generate all the videos in the `Code/videos/` Folder. This is where the combined RGB and segmentation map video is Stored.


## Navigation Approach


1. **Image Acquisition and Segmentation**
   - At each navigation step, the quadrotor captures an RGB image using the SplatRenderer.
   - This image is passed through a trained UNet segmentation model (Window_Segmentaion), producing a binary segmentation mask that highlights window regions.


2. **Window Detection and Alignment**
   - The segmentation mask is processed to detect all window candidates using contour analysis.
   - The largest detected window (by area) is selected as the navigation target, as it is the closest window in view.
   - The center of the detected window is compared to the center of the image to compute alignment errors (horizontal and vertical offsets).


3. **Iterative Alignment**
   - If the alignment error exceeds a predefined threshold (WINDOW_THRESHOLD), the quadrotor computes a corrective movement:
     - Lateral and vertical adjustments are calculated based on the pixel error, scaled to real-world coordinates.
     - The quadrotor moves incrementally to reduce the misalignment, capturing and segmenting new images at each step.
   - This process repeats until the window is sufficiently centered in the image (within threshold).


4. **Approach and Window Traversal**
   - Once aligned, the quadrotor estimates the distance to the window using either depth data or the window’s area in the image.
   - It then moves forward by the computed approach distance, passing through the window.
   - After traversal, the quadrotor updates its position and prepares to detect the next window.


5. **Multi-Window Navigation**
   - The above steps are repeated for each window in the course.
   - The system tracks the number of successfully traversed windows.
   - After passing through all required windows, the navigation loop terminates and the quadrotor stops.


6. **Frame Logging and Video Generation**
   - Throughout navigation, RGB and segmentation frames are periodically saved for debugging and visualization.
   - After the run, these frames are compiled into videos for analysis.


**Key Features:**
- Robust segmentation and window detection using UNet and OpenCV.
- Dynamic alignment and re-alignment based on real-time image feedback.
- Modular waypoint navigation with trajectory generation and PID control.

