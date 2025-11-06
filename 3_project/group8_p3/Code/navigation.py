import numpy as np
from scipy.integrate import solve_ivp
from pyquaternion import Quaternion
from control import QuadrotorController
from quad_dynamics import model_derivative
import tello
import cv2

def goToWaypoint(currentPose, targetPose, targetOrientation=None, velocity=0.1):
    """
    Navigate quadrotor to a target waypoint
    
    Parameters:
    - currentPose: Dictionary with keys:
        'position': [x, y, z] in NED frame (meters)
        'rpy': [roll, pitch, yaw] in radians
    - targetPose: Dictionary OR array
        If dict: {'position': [x,y,z], 'rpy': [r,p,y]} - control both
        If array: [x, y, z]
    - targetOrientation: Optional target orientation [roll, pitch, yaw] in radians
    - velocity: cruise velocity (m/s), default 0.1
    """
    
    # Handle both formats for backward compatibility
    if isinstance(targetPose, dict):
        target_position = np.array(targetPose['position'])
        target_orientation = np.array(targetPose['rpy'])
        control_orientation = True
    else:
        # Old format: just position
        target_position = np.array(targetPose)
        target_orientation = None
        control_orientation = False
    
    dt = 0.01  # 10ms timestep
    tolerance = 0.1  # 10cm tolerance
    max_time = 30.0  # Maximum 30 seconds
    
    # Initialize controller
    controller = QuadrotorController(tello)
    param = tello
    
    # Extract current state
    pos = np.array(currentPose['position'])
    rpy = np.array(currentPose['rpy'])  # roll, pitch, yaw in radians
    
    # Initialize velocities to zero (starting from hover)
    vel = np.zeros(3)
    pqr = np.zeros(3)
    
    # Convert roll, pitch, yaw to quaternion
    roll, pitch, yaw = rpy
    quat = Quaternion(axis=[0, 0, 1], radians=yaw) * \
           Quaternion(axis=[0, 1, 0], radians=pitch) * \
           Quaternion(axis=[1, 0, 0], radians=roll)
    
    # Build state vector [x, y, z, vx, vy, vz, qx, qy, qz, qw, p, q, r]
    current_state = np.concatenate([pos, vel, [quat.x, quat.y, quat.z, quat.w], pqr])
    
    target_position = np.array(targetPose)
    
    # Calculate distance and estimated time
    distance = np.linalg.norm(target_position - pos)
    estimated_time = min(distance / velocity * 2.0, max_time)
    
    print(f"  Navigating: {pos} → {target_position}")
    print(f"  Distance: {distance:.2f}m, Est. time: {estimated_time:.1f}s")
    
    # Check if already at target
    if distance < tolerance:
        print("  Already at target!")
        return {'position': pos, 'rpy': rpy}
    
    # Generate trajectory
    num_points = int(estimated_time / dt)
    time_points = np.linspace(0, estimated_time, num_points)
    
    # Create trajectory with trapezoidal velocity profile
    direction = target_position - pos
    unit_direction = direction / distance
    
    trajectory_points = []
    velocities = []
    accelerations = []
    
    accel_time = min(1.0, estimated_time * 0.25)
    decel_time = accel_time
    cruise_time = estimated_time - accel_time - decel_time
    
    cruise_vel = min(velocity, distance / (0.5 * accel_time + cruise_time + 0.5 * decel_time))
    
    for t in time_points:
        if t <= accel_time:
            # Acceleration phase
            vel_mag = (cruise_vel / accel_time) * t
            acc_mag = cruise_vel / accel_time
            progress = 0.5 * (cruise_vel / accel_time) * t * t / distance
        elif t <= accel_time + cruise_time:
            # Cruise phase
            vel_mag = cruise_vel
            acc_mag = 0.0
            progress = (0.5 * cruise_vel * accel_time + cruise_vel * (t - accel_time)) / distance
        else:
            # Deceleration phase
            t_decel = t - accel_time - cruise_time
            vel_mag = cruise_vel - (cruise_vel / decel_time) * t_decel
            acc_mag = -cruise_vel / decel_time
            progress = (0.5 * cruise_vel * accel_time + cruise_vel * cruise_time + 
                      cruise_vel * t_decel - 0.5 * (cruise_vel / decel_time) * t_decel * t_decel) / distance
        
        progress = np.clip(progress, 0.0, 1.0)
        position = pos + progress * direction
        vel_vec = vel_mag * unit_direction
        acc_vec = acc_mag * unit_direction
        
        trajectory_points.append(position)
        velocities.append(vel_vec)
        accelerations.append(acc_vec)
    
    trajectory_points = np.array(trajectory_points)
    velocities = np.array(velocities)
    accelerations = np.array(accelerations)
    
    # Set trajectory in controller
    controller.set_trajectory(trajectory_points, time_points, velocities, accelerations, target_rpy)
    
    # Simulation loop
    state = current_state.copy()
    
    for i, t in enumerate(time_points):
        # Compute control input
        control_input = controller.compute_control(state, t)
        
        # Check if reached
        current_pos = state[0:3]
        error = np.linalg.norm(current_pos - target_position)
        if error < tolerance and t > 1.0:
            print(f"  ✓ Reached at t={t:.2f}s, error={error:.3f}m")
            state_final = state
            break
        
        # Integrate dynamics
        if i < len(time_points) - 1:
            sol = solve_ivp(
                lambda t, X: model_derivative(t, X, control_input, param),
                [t, t + dt],
                state,
                method='RK45',
                max_step=dt
            )
            state = sol.y[:, -1]
            state_final = state
    else:
        # Loop completed without break
        state_final = state
        error = np.linalg.norm(state_final[0:3] - target_position)
        print(f"  Final error: {error:.3f}m")
    
    # Extract final pose
    final_pos = state_final[0:3]
    final_quat = Quaternion(state_final[9], state_final[6], state_final[7], state_final[8])  # w, x, y, z
    final_ypr = final_quat.yaw_pitch_roll  # Returns [yaw, pitch, roll]
    final_rpy = np.array([final_ypr[2], final_ypr[1], final_ypr[0]])  # [roll, pitch, yaw]
    
    newPose = {
        'position': final_pos,
        'rpy': final_rpy
    }
    
    return newPose

def navigate_through_window(renderer, currentPose, segmentor, detector, windowCount, 
                           max_iterations=10, step_distance=0.5):
    """
    Visual servoing approach to navigate through a window
    UPDATED: Saves images at EVERY iteration for debugging
    
    Parameters:
    - renderer: SplatRenderer instance
    - currentPose: Current drone pose dictionary
    - segmentor: Window segmentation model
    - detector: WindowDetector instance
    - windowCount: Current window number (for logging)
    - max_iterations: Maximum alignment iterations
    - step_distance: Distance to move forward each step (meters)
    
    Returns:
    - success: bool, whether window was successfully traversed
    - currentPose: Updated pose after navigation
    """
    
    print(f"\n{'='*60}")
    print(f"Window {windowCount + 1} - Visual Servoing Approach")
    print(f"{'='*60}")
    
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
        print(f"Current pose: position={currentPose['position']}, rpy={np.degrees(currentPose['rpy'])}")
        
        # Step 1: Render current view
        color_image, depth_image, metric_depth = renderer.render(
            currentPose['position'], 
            currentPose['rpy']
        )
        
        # Step 2: Segment windows
        segmented = segmentor.get_pred(color_image)
        segmented = cv2.normalize(segmented, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Step 3: Detect windows
        detections = detector.process_segmentation(segmented)
        
        # ============================================
        # SAVE ALL IMAGES AT EVERY ITERATION
        # ============================================
        iter_prefix = f'./log/window_{windowCount}_iter_{iteration:02d}'
        
        # Save RAW RGB (what drone sees)
        color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'{iter_prefix}_rgb.png', color_image_bgr)
        
        # Save segmentation
        cv2.imwrite(f'{iter_prefix}_segmentation.png', segmented)
        
        # Save detection visualization
        if detections:
            vis_img = detector.visualize_detection(color_image, detections, 
                                                   detector.get_closest_window(detections))
            cv2.imwrite(f'{iter_prefix}_detection.png', cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        else:
            # No detections - annotate RGB
            vis_img = color_image.copy()
            cv2.putText(vis_img, "NO WINDOWS DETECTED", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            cv2.imwrite(f'{iter_prefix}_detection.png', cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        
        # Save depth
        depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(f'{iter_prefix}_depth.png', depth_normalized)
        
        print(f"  💾 Saved: {iter_prefix}_*.png")
        
        # ============================================
        # PROCESS DETECTIONS
        # ============================================
        
        if not detections:
            print("⚠️  No windows detected!")
            
            # Recovery attempt - move forward slightly
            if iteration < max_iterations - 2:
                print("   Attempting recovery: moving forward 0.2m...")
                target = currentPose['position'].copy()
                target[0] += 0.2
                currentPose = goToWaypoint(currentPose, target, velocity=0.2)
                continue
            else:
                print("   Stopping.")
                return False, currentPose
        
        # Step 4: Select closest window
        closest_window = detector.get_closest_window(detections)
        
        # Step 5: Calculate stats
        error_x, error_y, error_mag = detector.calculate_alignment_error(closest_window)
        area_pct = closest_window['area'] / detector.image_area * 100
        
        print(f"Window Stats:")
        print(f"  Area: {area_pct:.1f}% of image")
        print(f"  Alignment error: {error_mag:.1f} pixels ({error_x:+.0f}x, {error_y:+.0f}y)")
        print(f"  Center: {closest_window['center']}")
        
        # Step 6: Check status
        is_aligned = detector.is_aligned(closest_window)
        is_close = detector.is_close_enough(closest_window)
        
        # Step 7: Decision logic
        if is_aligned and is_close:
            print(f"✅ ALIGNED & CLOSE - Flying through window!")
            
            # Final push through the window
            target = currentPose['position'].copy()
            target[0] += 2.0  # Move forward 2 meters to ensure we're through
            
            currentPose = goToWaypoint(currentPose, target, velocity=1.0)
            
            print(f"🎯 Successfully passed through window {windowCount + 1}!")
            return True, currentPose
        
        elif area_pct < 2.0:
            # Window too small/far - move forward with correction
            print(f"⚠️  Window small ({area_pct:.1f}%), correcting alignment while approaching...")
            
            target = detector.compute_navigation_target(
                currentPose['position'], 
                closest_window, 
                forward_distance=0.3  # Small steps when far
            )
            
            print(f"Moving to: {target}")
            currentPose = goToWaypoint(currentPose, target, velocity=0.2)
        
        elif is_aligned:
            print(f"→  Aligned but far - approaching...")
            distance_factor = detector.get_approach_distance(closest_window)
            move_distance = step_distance * distance_factor * 0.5  # Conservative
            
            # Move straight forward (already aligned)
            target = currentPose['position'].copy()
            target[0] += move_distance
            
            print(f"Moving to: {target}")
            currentPose = goToWaypoint(currentPose, target, velocity=0.3)
            
        else:
            print(f"⟲ Not aligned - correcting position...")
            
            # Compute target with lateral correction
            target = detector.compute_navigation_target(
                currentPose['position'], 
                closest_window, 
                forward_distance=step_distance * 0.4  # Slower while correcting
            )
            
            print(f"Moving to: {target}")
            currentPose = goToWaypoint(currentPose, target, velocity=0.3)
    
    # If we exhausted iterations without passing through
    print(f"⚠️  Max iterations reached without passing through window")
    return False, currentPose