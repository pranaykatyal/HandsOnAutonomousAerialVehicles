# Change Log

## [Date: 2025-12-10]

### Changes in `main.py`
- Updated proportional gain to `0.8` in the `visual_servo_to_window` function for better alignment control.
- Explicitly zeroed out forward movement during alignment to ensure sequential alignment logic.
- Added debugging statements to check for collisions at target positions.
- Introduced a `safe_update_position` method to centralize collision checking and ensure all position updates are validated before proceeding.
- Replaced direct position updates with calls to `safe_update_position` in `visual_servo_to_window` to integrate automatic collision handling.
- Added debug statements to verify focal lengths (`fx`, `fy`) and depth during visual servoing.
- Ensured consistent scaling in metric error calculation for pixel-to-metric conversion.
- Added a warning if focal lengths appear too small, suggesting a potential camera calibration issue.

### Changes in `control.py`
- Adjusted PID gains for position control:
  - Increased proportional gains (`kp`) for `x` and `y` to 1.2, and for `z` to 1.5.
- Adjusted PID gains for velocity control:
  - Reduced derivative gains (`kd`) for `vx` and `vy` to 0.3, and for `vz` to 0.4.

### Purpose of Changes
- Ensure collision checks are performed automatically before any position updates.
- Improve alignment accuracy and prevent unintended forward movement during alignment.
- Enhance debugging capabilities for collision detection.
- Improve stability and responsiveness of the drone's position and velocity control.
- Address scaling issues in visual servoing to prevent large movement errors.

### Next Steps
- Test the updated system to verify improvements in stability and scaling.
- Monitor debug outputs for focal lengths and depth to ensure proper calibration.

## [Date: 2025-12-11]

### Changes in `main.py`
- Ensured frames are saved during yaw alignment with proper numbering and overlay text displaying the current pose.
- Added detailed debug statements to confirm frame-saving during yaw alignment.
- Fixed the renderer usage to ensure the correct renderer is used for rendering frames during yaw alignment.
- Improved the logic to ensure frame-saving is triggered consistently during each iteration of the yaw alignment loop.

### Purpose of Changes
- Ensure proper visualization and debugging during yaw alignment.
- Fix issues with frame-saving logic to ensure all frames are captured and saved correctly.
- Use the correct renderer for accurate frame rendering.

### Next Steps
- Verify that all frames are saved correctly during yaw alignment.
- Test the updated system to ensure the renderer is functioning as expected.