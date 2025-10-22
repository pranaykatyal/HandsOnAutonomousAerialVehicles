# Evaluating window pose estimation using PnP

## Structure
```text
├── Code
│   ├── main.py                 # Main entry point - implement segmentNearestWindow() and navigation logic
│   ├── splat_render.py         # Gaussian splat renderer: generates RGB + depth from pose
│   ├── control.py              # PID controller and QuadrotorController class
│   ├── quad_dynamics.py        # Quadrotor dynamics simulation
│   ├── tello.py                # DJI Tello drone parameters
│   └── trajectory_generator.py # B-spline trajectory generation
├── data
│   ├── washburn-env6-itr0-1fps/           # Splat rendering checkpoint
│   └── washburn-env6-itr0-1fps_nf_format/ # COLMAP image dataset
├── render_settings
│   └── render_settings_2.json  # Render configuration (resolution, FOV, camera matrix)
└── log/                        # Controller logs (auto-created)
```
## How to run?
You should implement `segmentNearestWindow()` function in main.py and write code for `navigation`.


## Supporting Files

- **`splat_render.py`**: Renderer that takes in pose information and generates RGB and depth images
- **`control.py`**: PID controller implementation with QuadrotorController class
- **`quad_dynamics.py`**: Full 6-DOF quadrotor dynamics simulation
- **`tello.py`**: DJI Tello drone physical parameters (mass, inertia, rotor positions)
- **`trajectory_generator.py`**: B-spline trajectory generation utilities
- **`data/`**: Contains two datasets:
  - `washburn-env6-itr0-1fps/`: Splat rendering checkpoint
  - `washburn-env6-itr0-1fps_nf_format/`: COLMAP-formatted dataset

## Navigation Approach

You have full flexibility in implementing the navigation:

- **Navigation Function Provided**: A `goToWaypoint()` function is already implemented to reach any target waypoint
- **Customizable Parameters**: You are free to modify parameters in `main.py`:
  - Robot speed (`velocity` parameter in `goToWaypoint()`)
  - Navigation tolerance
  - Maximum time limits
  - Trajectory profile (acceleration/deceleration times)
- **Controller Tuning**: Modify PID gains, angular rate limits, and other controller parameters in `control.py` if needed
- **Your Task**: 
  1. Implement `segmentNearestWindow()` to detect windows
  2. Compute target waypoints from detected windows (using mask + depth)
  3. Use the provided `goToWaypoint()` function to navigate through windows
  4. Feel free to use any approach or algorithm for window detection and waypoint computation

The navigation system is modular - you can focus on the vision/planning aspects while the low-level control is handled automatically.

## Deliverables

### 1. Navigation Video
Create a video showing the drone's navigation:
- **Left side:** FPV camera view (RGB image)
- **Right side:** Segmented window visualization (overlay mask when window is detected)
- Save as MP4 format

### 2. Robot Path Plot
Create a 3D matplotlib plot showing:
- Complete robot trajectory through all windows
- Waypoint markers
- Axis labels in NED frame (X, Y, Z in meters)
- Save as PNG/PDF

Submit both the video and the trajectory plot along with your code.