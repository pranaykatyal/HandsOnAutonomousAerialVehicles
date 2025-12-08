# Drone Racing !!!!!!!!!!!!!!!!

## Structure
```text
├── Code
│   ├── main.py                 # Main entry point - implement segmentNearestWindow() and navigation logic
│   ├── splat_render.py         # Gaussian splat renderer: generates RGB + depth from pose
│   ├── control.py              # PID controller and QuadrotorController class
│   ├── quad_dynamics.py        # Quadrotor dynamics simulation
│   ├── collisionChecker.py        # collision checker
│   ├── tello.py                # DJI Tello drone parameters
│   └── trajectory_generator.py # B-spline trajectory generation
├── data
│   ├── P5_colmap_splat/           # Splat rendering checkpoint
│   └── P5_colmap/                 # COLMAP image dataset
├── render_settings
│   └── render_settings.json  # Render configuration (resolution, FOV, camera matrix)
```
## How to run?
- python `main.py`


## Supporting Files

- **`splat_render.py`**: Renderer that takes in pose information and generates RGB and depth images
- **`control.py`**: PID controller implementation with QuadrotorController class
- **`quad_dynamics.py`**: Full 6-DOF quadrotor dynamics simulation
- **`collisionChecker.py`**: Check for collision of a point
- **`tello.py`**: DJI Tello drone physical parameters (mass, inertia, rotor positions)
- **`trajectory_generator.py`**: B-spline trajectory generation utilities
- **`data/`**: Contains two datasets:
  - `P5_colmap_splat/`: Splat rendering checkpoint
  - `P5_colmap/`: COLMAP-formatted dataset

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
  1. Navigate through the windows and comeback (again through the windows in reverse order to start position) as fast as you can, without hitting any obstacle.
  2. Make sure you check for collision (`doesItCollide(pos)` returns a `True` or `False`)
  3. USING GROUND-TRUTH DEPTH IS **NOT** ALLOWED.
  

The navigation system is modular - you can focus on the vision/planning aspects while the low-level control is handled automatically.

## Deliverables

### Navigation Video
Create a video showing the drone's navigation:
- **Left side:** FPV camera view (RGB image)
- **Right side:** Segmented window visualization (overlay mask when window is detected)
- Save as MP4 format
