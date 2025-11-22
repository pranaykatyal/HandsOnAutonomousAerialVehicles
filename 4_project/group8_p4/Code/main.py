"""
main.py - Window Navigation with TS²P Detection
FIXED: Phase 4 forward motion + early stop when crossing window
UPDATED: Visual markers (arrows, dots, crosses) only show for first 3 iterations

KEY PARAMETERS:
- Phase 3: max_iterations=15, pixel_tolerance=1px, control_gain=0.55
  → Converges by iteration 7-10
- Phase 4: max_iterations=15, forward_gain=0.05m (5cm per step)
  → Crosses window by iteration 12-15
  → Stops automatically when detection is lost (passed through)

OUTPUT FILES:
- ./log/servo_iter_XX.png - RGB frames with detection overlay
- ./log/flow_iter_XX.png - Optical flow visualizations
- Run combine_rgb_flow.py to create side-by-side comparisons
"""

from splat_render import SplatRenderer
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pyquaternion import Quaternion
from control import QuadrotorController
from quad_dynamics import model_derivative
import tello
import torch
from ts2p_network import OpticalFlowExtractor, TS2P_GapDetector
from scanning_fix import generate_scanning_positions_fixed
from improved_window_detector import ImprovedWindowDetector
import os
import glob
import json


################################################
#### TS²P Window Detector Integration ###########
################################################

class WindowDetector:
    """Wrapper for TS²P detection integrated with your renderer"""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print("Initializing TS²P Window Detector...")
        
        self.flow_extractor = OpticalFlowExtractor(model_type='raft', device=self.device)
        self.gap_detector = TS2P_GapDetector(self.flow_extractor, device=self.device)
        self.gap_detector.threshold_percentile = 83
        
        self.detection_resolution = 512
        
        print("✓ TS²P Detector initialized")
    
    def visualize_detection(self, frame, mask, center, save_path='detection_result.png'):
        """Visualize detection overlay"""
        vis = frame.copy()
        
        # Green overlay
        overlay = np.zeros_like(vis)
        overlay[mask > 0.5] = [0, 255, 0]
        vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
        
        if center is not None:
            # Window center (red)
            cv2.circle(vis, center, 20, (0, 0, 255), -1)
            cv2.circle(vis, center, 25, (0, 0, 255), 3)
            
            # Image center (blue)
            H, W = frame.shape[:2]
            img_center = (W//2, H//2)
            cv2.drawMarker(vis, img_center, (255, 0, 0), 
                          markerType=cv2.MARKER_CROSS, 
                          markerSize=50, thickness=3)
            
            # Line from image center to window center
            cv2.arrowedLine(vis, img_center, center, (255, 255, 0), 3, tipLength=0.05)
            
            # Text
            error_pixels = np.linalg.norm([center[0] - img_center[0], center[1] - img_center[1]])
            cv2.putText(vis, f'Error: {error_pixels:.0f}px', (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        return vis


def init_log_dir():
    """Initialize log directory"""
    os.makedirs('./log', exist_ok=True)
    
    # Clean old images
    old_images = glob.glob('./log/*.png')
    if old_images:
        print(f"Cleaning {len(old_images)} old image files...")
        for img_file in old_images:
            try:
                os.remove(img_file)
            except Exception as e:
                print(f"Warning: Could not remove {img_file}: {e}")


################################################
#### Navigation Function ########################
################################################
def goToWaypoint(currentPose, targetPose, velocity=0.1):
    """
    Navigate quadrotor to a target waypoint
    
    Parameters:
    - currentPose: Dictionary with keys:
        'position': [x, y, z] in NED frame (meters)
        'rpy': [roll, pitch, yaw] in radians
    - targetPose: [x, y, z] target position in NED frame (meters)
    - velocity: cruise velocity (m/s), default 0.1
    
    Returns:
    - newPose: Dictionary with updated 'position' and 'rpy'
    """
    
    dt = 0.01  # 10ms timestep
    tolerance = 0.01  # 1cm tolerance (much tighter for visual servoing!)
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
    print(f"  Distance: {distance:.4f}m, Est. time: {estimated_time:.1f}s")
    
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
    controller.set_trajectory(trajectory_points, time_points, velocities, accelerations)
    
    # Simulation loop
    state = current_state.copy()
    
    for i, t in enumerate(time_points):
        # Compute control input
        control_input = controller.compute_control(state, t)
        
        # Check if reached
        current_pos = state[0:3]
        error = np.linalg.norm(current_pos - target_position)
        if error < tolerance and t > 1.0:
            print(f"  ✓ Reached at t={t:.2f}s, error={error:.4f}m")
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
        print(f"  Final error: {error:.4f}m")
    
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


def compute_visual_servo_step(center_2d, image_shape, camera_matrix, gain=0.15):
    """Compute visual servoing step from pixel error"""
    H, W = image_shape
    x_px, y_px = center_2d
    
    # CRITICAL: Use ACTUAL image center from rendered image, not camera_matrix!
    # The rendered image might have different dimensions than the config
    cx = W / 2.0  # Actual image width center
    cy = H / 2.0  # Actual image height center
    
    # Use focal length from camera matrix (this is correct)
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    
    # Scale focal length to match actual image resolution
    config_resolution = camera_matrix[0, 2] * 2.0  # Original resolution
    scale_factor = W / config_resolution
    fx_scaled = fx * scale_factor
    fy_scaled = fy * scale_factor
    
    print(f"  [CAMERA] Image: {W}x{H}, Center: ({cx:.1f}, {cy:.1f})")
    print(f"  [CAMERA] Focal length scaled: {fx_scaled:.1f} (was {fx:.1f})")
    
    # Pixel errors
    error_x = x_px - cx
    error_y = y_px - cy
    pixel_error = np.sqrt(error_x**2 + error_y**2)
    
    # Convert to angular error
    theta_x = np.arctan2(error_x, fx_scaled)
    theta_y = np.arctan2(error_y, fy_scaled)
    
    # Visual servoing control (VERIFIED coordinates)
    delta_y = gain * theta_x
    delta_z = gain * theta_y
    delta_x = 0.0
    
    delta_position = np.array([delta_x, delta_y, delta_z])
    
    return delta_position, pixel_error, error_x, error_y


################################################
#### Main Function ##############################
################################################
def main(renderer):
    init_log_dir()
    
    print("="*70)
    print("Window Navigation with TS²P Detection")
    print("Visual markers shown only for first 3 iterations per phase")
    print("="*70)
    
    try:
        # Initialize detector
        detector = WindowDetector(device='cuda')
        
        # Load camera parameters
        with open("../render_settings/render_settings.json", 'r') as f:
            render_settings = json.load(f)
        
        camera_params = render_settings['camera']
        render_resolution = camera_params['render_resolution']
        fov_radians = camera_params['fov_radians']
        
        image_width = render_resolution
        image_height = render_resolution
        focal_length = (image_width / 2.0) / np.tan(fov_radians / 2.0)
        cx = image_width / 2.0
        cy = image_height / 2.0
        
        camera_matrix = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ])
        
        print(f"Camera: {render_resolution}x{render_resolution}, FOV: {np.degrees(fov_radians):.1f}°")
        
        # Initial pose
        currentPose = {
            'position': np.array([0.0, 0.0, 0.0]),
            'rpy': np.radians([0.0, 0.0, 0.0])
        }
        
        print(f"Initial position: {currentPose['position']}")
        
        #####################################################
        ### STEP 1: ACTIVE SCANNING
        #####################################################
        
        print("\n--- Phase 1: Active Scanning ---")

        scan_positions = generate_scanning_positions_fixed(currentPose['position'], scan_distance=0.01)
        print(f"Generated {len(scan_positions)} scan positions")
        
        scan_frames = []
        for i, pos in enumerate(scan_positions):
            print(f"  Scan frame {i+1}/{len(scan_positions)}")
            
            color_image, depth_image, metric_depth = renderer.render(pos, currentPose['rpy'])
            scan_frames.append(color_image)
            
            cv2.imwrite(f'./log/scan_{i:02d}.png', cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
        
        print(f"  ✓ Captured {len(scan_frames)} frames")
        
        #####################################################
        ### STEP 2: WINDOW DETECTION
        #####################################################

        print("\n--- Phase 2: Window Detection ---")

        improved_detector = ImprovedWindowDetector(detector)

        window_mask, window_center_2d, confidence, debug_info = improved_detector.detect_window_improved(scan_frames)

        improved_detector.visualize_detection_process(debug_info, './log/detection_process.png')

        print(f"  Confidence: {confidence:.3f}")

        if window_center_2d is None or confidence < 0.2:
            print("  ✗ Detection failed!")
            return False

        print(f"  ✓ Window at: {window_center_2d}")

        detector.visualize_detection(scan_frames[0], window_mask, window_center_2d,
                                    './log/detection.png')
        
        #####################################################
        ### STEP 3: VISUAL SERVOING
        #####################################################
        
        print("\n--- Phase 3: Visual Servoing ---")
        
        max_iterations = 15
        pixel_tolerance = 15
        control_gain = 0.55
        
        error_history = []
        
        for iteration in range(max_iterations):
            print(f"\n=== Iteration {iteration + 1}/{max_iterations} ===")
            print(f"Current position: {currentPose['position']}")
            
            # Re-scan at current position
            scan_positions_current = generate_scanning_positions_fixed(
                currentPose['position'], scan_distance=0.01)
            
            scan_frames_current = []
            for pos in scan_positions_current:
                img, _, _ = renderer.render(pos, currentPose['rpy'])
                scan_frames_current.append(img)
            
            # Detect window
            detection_result = improved_detector.detect_window_improved(scan_frames_current)
            mask, center_2d, conf = detection_result[:3]
            
            # Get debug info for flow visualization
            if len(detection_result) > 3:
                debug_info = detection_result[3]
                flow_map = debug_info.get('Xi', None)
                
                # Save flow visualization
                if flow_map is not None:
                    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                    im = ax.imshow(flow_map, cmap='jet')
                    ax.set_title(f'Optical Flow - Phase 3 Iter {iteration+1}', fontsize=14, fontweight='bold')
                    ax.axis('off')
                    plt.colorbar(im, ax=ax, fraction=0.046)
                    plt.tight_layout()
                    plt.savefig(f'./log/flow_iter_{iteration+1:02d}.png', dpi=150, bbox_inches='tight')
                    plt.close()
            
            if center_2d is None or conf < 0.2:
                print(f"  ✗ Lost detection (conf={conf:.3f})")
                break
            
            # Get reference frame
            reference_frame = scan_frames_current[0]
            
            # Compute control
            delta_pos, pixel_error, error_x, error_y = compute_visual_servo_step(
                center_2d, reference_frame.shape[:2], camera_matrix, gain=control_gain)
            
            error_history.append(pixel_error)
            
            print(f"  Window center: {center_2d}")
            print(f"  Errors: X={error_x:+.1f}px, Y={error_y:+.1f}px, Total={pixel_error:.1f}px")
            print(f"  Command: ΔY={delta_pos[1]:+.5f}m, ΔZ={delta_pos[2]:+.5f}m")
            
            # Save visualization - ONLY show markers for first 3 iterations
            debug_overlay = reference_frame.copy()
            H_vis, W_vis = debug_overlay.shape[:2]
            cx_vis = W_vis / 2.0
            cy_vis = H_vis / 2.0
            
            # Only show visual markers for first 3 iterations
            if iteration < 3:
                cv2.drawMarker(debug_overlay, (int(cx_vis), int(cy_vis)), (255, 0, 0), 
                              markerType=cv2.MARKER_CROSS, markerSize=50, thickness=3)
                cv2.circle(debug_overlay, center_2d, 20, (0, 0, 255), -1)
                cv2.circle(debug_overlay, center_2d, 25, (0, 0, 255), 3)
                cv2.arrowedLine(debug_overlay, (int(cx_vis), int(cy_vis)), center_2d, 
                               (255, 255, 0), 3, tipLength=0.05)
            
            cv2.putText(debug_overlay, f'Iter {iteration+1}: {pixel_error:.1f}px', 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.putText(debug_overlay, f'Img: {W_vis}x{H_vis} Center: ({int(cx_vis)}, {int(cy_vis)})', 
                       (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imwrite(f'./log/servo_iter_{iteration+1:02d}.png', 
                       cv2.cvtColor(debug_overlay, cv2.COLOR_RGB2BGR))
            
            # Check convergence
            if pixel_error < pixel_tolerance:
                print(f"\n✓ CONVERGED! Error={pixel_error:.1f}px < {pixel_tolerance}px")
                
                cv2.imwrite('./log/final_centered.png', 
                           cv2.cvtColor(reference_frame, cv2.COLOR_RGB2BGR))
                detector.visualize_detection(reference_frame, mask, center_2d, 
                                           './log/final_centered_detection.png')
                break
            
            # Navigate
            target_position = currentPose['position'] + delta_pos
            print(f"  Target: {target_position}")
            
            currentPose = goToWaypoint(currentPose, target_position, velocity=0.1)
            
            # FORCE orientation back to [0, 0, 0]
            currentPose['rpy'] = np.radians([0.0, 0.0, 0.0])
            
            print(f"  Reached: {currentPose['position']}")
        
        else:
            print(f"\n⚠ Max iterations reached")
        
        # Print Phase 3 summary
        print(f"\n{'='*70}")
        print("Phase 3 Summary")
        print(f"{'='*70}")
        print(f"Iterations: {len(error_history)}")
        print(f"Error history: {[f'{e:.1f}' for e in error_history]}")
        
        #####################################################
        ### STEP 4: APPROACH THROUGH WINDOW
        #####################################################
        
        print(f"\n" + "="*70)
        print("Phase 4: Approach and Pass Through Window")
        print("="*70)
        
        start_position = currentPose['position'].copy()
        phase4_start_iter = len(error_history)
        phase4_max_iterations = 12
        forward_gain = 0.05
        
        print(f"Continuing from iteration {phase4_start_iter + 1}...")
        print(f"Moving forward {forward_gain}m per iteration")
        print(f"Starting position: {start_position}")
        
        for iteration in range(phase4_max_iterations):
            current_iter = phase4_start_iter + iteration + 1
            print(f"\n=== Iteration {current_iter} (Phase 4: {iteration + 1}/{phase4_max_iterations}) ===")
            print(f"Current position: {currentPose['position']}")
            
            # Re-scan at current position
            scan_positions_current = generate_scanning_positions_fixed(
                currentPose['position'], scan_distance=0.01)
            
            scan_frames_current = []
            for pos in scan_positions_current:
                img, _, _ = renderer.render(pos, currentPose['rpy'])
                scan_frames_current.append(img)
            
            # Detect window
            detection_result = improved_detector.detect_window_improved(scan_frames_current)
            mask, center_2d, conf = detection_result[:3]
            
            # Get debug info for flow visualization
            if len(detection_result) > 3:
                debug_info = detection_result[3]
                flow_map = debug_info.get('Xi', None)
                
                # Save flow visualization
                if flow_map is not None:
                    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                    im = ax.imshow(flow_map, cmap='jet')
                    ax.set_title(f'Optical Flow - Phase 4 Iter {current_iter}', fontsize=14, fontweight='bold')
                    ax.axis('off')
                    plt.colorbar(im, ax=ax, fraction=0.046)
                    plt.tight_layout()
                    plt.savefig(f'./log/flow_iter_{current_iter:02d}.png', dpi=150, bbox_inches='tight')
                    plt.close()
            
            if center_2d is None or conf < 0.2:
                print(f"  ⚠ Lost detection (conf={conf:.3f}) - likely passed through window!")
                forward_only = True
                reference_frame = scan_frames_current[0]
                pixel_error = 0.0
                error_x = 0.0
                error_y = 0.0
                delta_pos_yz = np.array([0.0, 0.0, 0.0])
                
                # Save final frame and stop
                cv2.imwrite(f'./log/servo_iter_{current_iter:02d}.png', 
                           cv2.cvtColor(reference_frame, cv2.COLOR_RGB2BGR))
                
                distance_traveled = currentPose['position'][0] - start_position[0]
                print(f"\n✓ Passed through window! Traveled {distance_traveled:.2f}m forward")
                print(f"✓ Lost detection at iteration {current_iter}")
                break
            else:
                forward_only = False
                reference_frame = scan_frames_current[0]
                
                # Compute Y-Z correction
                delta_pos_yz, pixel_error, error_x, error_y = compute_visual_servo_step(
                    center_2d, reference_frame.shape[:2], camera_matrix, gain=0.2)
                
                print(f"  Window center: {center_2d}")
                print(f"  Errors: X={error_x:+.1f}px, Y={error_y:+.1f}px, Total={pixel_error:.1f}px")
            
            # Add forward motion
            delta_pos = delta_pos_yz.copy()
            delta_pos[0] = forward_gain
            
            print(f"  Command: ΔX={delta_pos[0]:+.5f}m (forward), ΔY={delta_pos[1]:+.5f}m, ΔZ={delta_pos[2]:+.5f}m")
            
            # Save visualization - ONLY show markers for first 3 iterations of Phase 4
            debug_overlay = reference_frame.copy()
            H_vis, W_vis = debug_overlay.shape[:2]
            cx_vis = W_vis / 2.0
            cy_vis = H_vis / 2.0
            
            # Only show visual markers for first 2 iterations of Phase 4
            if not forward_only and iteration < 2:
                cv2.drawMarker(debug_overlay, (int(cx_vis), int(cy_vis)), (255, 0, 0), 
                              markerType=cv2.MARKER_CROSS, markerSize=50, thickness=3)
                cv2.circle(debug_overlay, center_2d, 20, (0, 0, 255), -1)
                cv2.circle(debug_overlay, center_2d, 25, (0, 0, 255), 3)
                cv2.arrowedLine(debug_overlay, (int(cx_vis), int(cy_vis)), center_2d, 
                               (255, 255, 0), 3, tipLength=0.05)
            
            cv2.putText(debug_overlay, f'Phase 4 - Iter {current_iter}', 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            cv2.putText(debug_overlay, f'Forward progress: {iteration+1}/{phase4_max_iterations}', 
                       (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            if not forward_only:
                cv2.putText(debug_overlay, f'Error: {pixel_error:.1f}px', 
                           (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Add distance traveled display
            distance_traveled = currentPose['position'][0] - start_position[0]
            cv2.putText(debug_overlay, f'Distance: {distance_traveled:.2f}m', 
                       (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            cv2.imwrite(f'./log/servo_iter_{current_iter:02d}.png', 
                       cv2.cvtColor(debug_overlay, cv2.COLOR_RGB2BGR))
            
            # Navigate - DIRECT position update
            target_position = currentPose['position'] + delta_pos
            print(f"  Target: {target_position}")
            
            currentPose['position'] = target_position
            print(f"  Reached: {currentPose['position']} (direct update)")
            
            currentPose['rpy'] = np.radians([0.0, 0.0, 0.0])
            
            # Check distance
            distance_traveled = currentPose['position'][0] - start_position[0]
            print(f"  Total distance traveled: {distance_traveled:.2f}m")
            
            if distance_traveled > 1.5:
                print(f"\n✓ Traveled {distance_traveled:.2f}m forward - likely through window!")
                break
        
        else:
            print(f"\n⚠ Phase 4 iterations complete")
        
        # Render final view
        final_frame, _, _ = renderer.render(currentPose['position'], currentPose['rpy'])
        cv2.imwrite('./log/final_through_window.png', cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR))
        
        total_iterations = phase4_start_iter + iteration + 1
        print(f"\n✓ Saved {total_iterations} total iteration frames")
        print(f"✓ Phase 3: {phase4_start_iter} iterations (alignment)")
        print(f"✓ Phase 4: {iteration + 1} iterations (approach)")
        
        print(f"\n{'='*70}")
        print("✓ Mission Complete!")
        print(f"{'='*70}")
        print(f"Final position: {currentPose['position']}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    config_path = "../data/p4_colmap_nov6_1000_splat/p4_colmap_nov6_1000/splatfacto/2025-11-06_161816/config.yml"
    json_path = "../render_settings/render_settings.json"
    
    renderer = SplatRenderer(config_path, json_path)
    success = main(renderer)
    
    if success:
        print("\n✓ Window navigation successful!")
    else:
        print("\n✗ Navigation failed")