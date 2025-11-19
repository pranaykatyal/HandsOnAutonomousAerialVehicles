"""
main.py - Window Navigation with TSÂ²P Detection
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
import os
import glob
import json


################################################
#### TSÂ²P Window Detector Integration ###########
################################################

class WindowDetector:
    """Wrapper for TSÂ²P detection integrated with your renderer"""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print("Initializing TSÂ²P Window Detector...")
        
        self.flow_extractor = OpticalFlowExtractor(model_type='raft', device=self.device)
        self.gap_detector = TS2P_GapDetector(self.flow_extractor, device=self.device)
        self.gap_detector.threshold_percentile = 83  # Optimized threshold
        
        # Scanning parameters - GAUSSIAN SPLAT COORDINATES
        # Scene is normalized: ±0.2 splat units (0.4 total)
        # 0.05 scan = 12.5% of scene = EXCELLENT for parallax
        self.num_scan_frames = 5
        self.scan_distance = 0.05  # splat units (NOT meters!)
        self.scan_step = self.scan_distance / (self.num_scan_frames - 1)
        
        # Target resolution for detection (to avoid OOM with 2064x2064 images)
        self.detection_resolution = 512  # Downsample for detection
        
        print("âœ“ TSÂ²P Detector initialized (IoU: 0.69)")
        print(f"  Detection resolution: {self.detection_resolution}x{self.detection_resolution}")
    
    def generate_scanning_positions(self, current_position):
        """Generate diagonal scanning trajectory positions"""
        positions = []
        positions.append(current_position.copy())
        
        for i in range(1, self.num_scan_frames):
            progress = i / (self.num_scan_frames - 1)
            offset_x = -progress * self.scan_distance / np.sqrt(2)
            offset_y = progress * self.scan_distance / np.sqrt(2)
            offset_z = progress * self.scan_distance / np.sqrt(2)
            
            scan_pos = current_position.copy()
            # scan_pos[0] += offset_x
            scan_pos[1] += offset_y * 2.0 # Scale to match splat units
            scan_pos[2] += offset_z * 2.0 # Scale to match splat units 
            positions.append(scan_pos)
        
        return positions
    
    def detect_window(self, frames):
        """
        Detect window from scanning sequence with downsampling
        
        Args:
            frames: List of (H, W, 3) RGB numpy arrays
        Returns:
            mask: (H, W) binary mask at ORIGINAL resolution
            center: (x, y) at ORIGINAL resolution
            confidence: [0, 1]
        """
        # Get original resolution
        original_H, original_W = frames[0].shape[:2]
        print(f"  Original frame size: {original_W}x{original_H}")
        
        # FILTER OUT GREEN SCREEN (bright green artifacts at edges)
        filtered_frames = []
        for frame in frames:
            # Create mask for green screen (very bright green)
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            lower_green = np.array([40, 40, 40])  # Greenish hue, some saturation
            upper_green = np.array([80, 255, 255])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # Also check for very bright green in RGB
            green_bright = (frame[:, :, 1] > 200) & (frame[:, :, 0] < 150) & (frame[:, :, 2] < 150)
            green_mask = green_mask | (green_bright.astype(np.uint8) * 255)
            
            # Black out green screen regions
            frame_filtered = frame.copy()
            frame_filtered[green_mask > 0] = [0, 0, 0]
            filtered_frames.append(frame_filtered)
        
        print(f"  Filtered green screen artifacts")
        
        # Downsample frames to avoid OOM
        downsampled_frames = []
        for frame in filtered_frames:
            frame_small = cv2.resize(frame, 
                                    (self.detection_resolution, self.detection_resolution),
                                    interpolation=cv2.INTER_LINEAR)
            downsampled_frames.append(frame_small)
        
        print(f"  Downsampled to: {self.detection_resolution}x{self.detection_resolution}")
        
        # Convert to torch tensor
        frames_np = np.array(downsampled_frames)
        frames_tensor = torch.from_numpy(frames_np).float() / 255.0
        frames_tensor = frames_tensor.permute(0, 3, 1, 2)
        frames_tensor = frames_tensor.unsqueeze(0).to(self.device)
        
        # Run TSÂ²P detection
        print(f"  Running TSÂ²P detection...")
        with torch.no_grad():
            mask_tensor, flow_magnitude = self.gap_detector.detect_gap(frames_tensor, refine=True)
        
        # Convert to numpy (downsampled)
        mask_small = mask_tensor[0, 0].cpu().numpy()
        
        # Upsample back to original resolution
        mask = cv2.resize(mask_small, (original_W, original_H), 
                         interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0.5).astype(np.float32)
        
        # Compute center at original resolution
        if mask.sum() > 100:
            y_coords, x_coords = np.where(mask > 0.5)
            center_x = int(x_coords.mean())
            center_y = int(y_coords.mean())
            center = (center_x, center_y)
            
            mask_area = mask.sum()
            confidence = min(1.0, mask_area / (original_H * original_W * 0.5))
        else:
            center = None
            confidence = 0.0
        
        return mask, center, confidence
    
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
    
    def detect_window_with_flow_debug(self, frames):
        """
        Detect window WITH complete flow visualization for debugging
        """
        import matplotlib.pyplot as plt
        
        # Get original resolution
        original_H, original_W = frames[0].shape[:2]
        print(f"  Original frame size: {original_W}x{original_H}")
        
        # Downsample frames
        downsampled_frames = []
        for frame in frames:
            frame_small = cv2.resize(frame, 
                                    (self.detection_resolution, self.detection_resolution),
                                    interpolation=cv2.INTER_LINEAR)
            downsampled_frames.append(frame_small)
        
        print(f"  Downsampled to: {self.detection_resolution}x{self.detection_resolution}")
        
        # Convert to torch tensor
        frames_np = np.array(downsampled_frames)
        frames_tensor = torch.from_numpy(frames_np).float() / 255.0
        frames_tensor = frames_tensor.permute(0, 3, 1, 2)
        frames_tensor = frames_tensor.unsqueeze(0).to(self.device)
        
        print(f"  Running TS²P detection...")
        
        # Get flow stack and Xi (flow magnitude)
        with torch.no_grad():
            # Compute flows
            F_stack = self.gap_detector.compute_temporal_flow_stack(frames_tensor)
            Xi = self.gap_detector.compute_flow_magnitude(F_stack)
            
            # Get threshold value
            B, _, H, W = Xi.shape
            threshold = torch.quantile(Xi.reshape(B, -1), 
                                    q=self.gap_detector.threshold_percentile/100.0, 
                                    dim=1, keepdim=True)
            threshold_val = threshold.item()
            
            # Apply threshold (inverted for openings)
            chi = (Xi > threshold).float()
            
            # Refine
            chi_refined = self.gap_detector.morphological_refinement(chi)
        
        # Convert to numpy for visualization
        Xi_np = Xi[0, 0].cpu().numpy()
        chi_np = chi[0, 0].cpu().numpy()
        chi_refined_np = chi_refined[0, 0].cpu().numpy()
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        
        # Row 1: Input frames
        axes[0, 0].imshow(downsampled_frames[0])
        axes[0, 0].set_title('Frame 0 (Reference)', fontsize=12)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(downsampled_frames[2])
        axes[0, 1].set_title('Frame 2 (Middle)', fontsize=12)
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(downsampled_frames[4])
        axes[0, 2].set_title('Frame 4 (Last)', fontsize=12)
        axes[0, 2].axis('off')
        
        # Row 2: Flow analysis
        im1 = axes[1, 0].imshow(Xi_np, cmap='jet')
        axes[1, 0].set_title(f'Flow Magnitude (Ξ)\nRange: [{Xi_np.min():.1f}, {Xi_np.max():.1f}]', fontsize=12)
        axes[1, 0].axis('off')
        plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
        
        # Histogram of flow values
        axes[1, 1].hist(Xi_np.flatten(), bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(threshold_val, color='r', linestyle='--', linewidth=2, 
                        label=f'Threshold ({self.gap_detector.threshold_percentile}%): {threshold_val:.1f}')
        axes[1, 1].set_xlabel('Flow Magnitude', fontsize=10)
        axes[1, 1].set_ylabel('Pixel Count', fontsize=10)
        axes[1, 1].set_title('Flow Distribution', fontsize=12)
        axes[1, 1].legend(fontsize=9)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Show both old and new logic
        mask_old = (Xi_np < threshold_val).astype(np.uint8) * 255
        axes[1, 2].imshow(mask_old, cmap='gray')
        axes[1, 2].set_title(f'OLD Logic: Ξ < {threshold_val:.1f}\n(LOW flow - WRONG)', 
                            fontsize=12, color='red')
        axes[1, 2].axis('off')
        
        # Row 3: Detection results
        axes[2, 0].imshow(chi_np, cmap='gray')
        axes[2, 0].set_title(f'NEW Logic: Ξ > {threshold_val:.1f}\n(HIGH flow - CORRECT)', 
                            fontsize=12, color='green')
        axes[2, 0].axis('off')
        
        axes[2, 1].imshow(chi_refined_np, cmap='gray')
        axes[2, 1].set_title(f'After Morphology\n{np.sum(chi_refined_np > 0.5):.0f} pixels', fontsize=12)
        axes[2, 1].axis('off')
        
        # Overlay on reference frame
        overlay = downsampled_frames[0].copy()
        mask_overlay = np.zeros_like(overlay)
        mask_overlay[chi_refined_np > 0.5] = [0, 255, 0]
        overlay = cv2.addWeighted(overlay, 0.7, mask_overlay, 0.3, 0)
        axes[2, 2].imshow(overlay)
        axes[2, 2].set_title('Detection Overlay', fontsize=12)
        axes[2, 2].axis('off')
        
        plt.suptitle('TS²P Detection Debug Visualization', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('./log/flow_debug.png', dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved flow debug to ./log/flow_debug.png")
        
        # Print detailed statistics
        print(f"\n  Flow Statistics:")
        print(f"    Min flow: {Xi_np.min():.2f}")
        print(f"    Max flow: {Xi_np.max():.2f}")
        print(f"    Mean flow: {Xi_np.mean():.2f}")
        print(f"    Median flow: {np.median(Xi_np):.2f}")
        print(f"    Threshold ({self.gap_detector.threshold_percentile}%): {threshold_val:.2f}")
        print(f"    Pixels with HIGH flow (>thresh): {np.sum(chi_np > 0.5):.0f} ({np.sum(chi_np > 0.5)/chi_np.size*100:.1f}%)")
        print(f"    After morphology: {np.sum(chi_refined_np > 0.5):.0f} pixels")
        
        # Continue with normal detection
        mask_tensor = chi_refined
        
        # Convert to numpy (downsampled)
        mask_small = mask_tensor[0, 0].cpu().numpy()
        
        # Upsample back to original resolution
        mask = cv2.resize(mask_small, (original_W, original_H), 
                        interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0.5).astype(np.float32)
        
        # Compute center at original resolution
        if mask.sum() > 100:
            y_coords, x_coords = np.where(mask > 0.5)
            center_x = int(x_coords.mean())
            center_y = int(y_coords.mean())
            center = (center_x, center_y)
            
            mask_area = mask.sum()
            confidence = min(1.0, mask_area / (original_H * original_W * 0.5))
        else:
            center = None
            confidence = 0.0
        
        return mask, center, confidence


def compute_window_3d_position(center_2d, depth_image, camera_matrix):
    """Compute 3D position from 2D pixel and depth"""
    x_px, y_px = center_2d
    
    # Get depth at window center (median around center)
    window_size = 20
    y_min = max(0, y_px - window_size)
    y_max = min(depth_image.shape[0], y_px + window_size)
    x_min = max(0, x_px - window_size)
    x_max = min(depth_image.shape[1], x_px + window_size)
    
    depth_window = depth_image[y_min:y_max, x_min:x_max]
    depth = np.median(depth_window)
    
    # Unproject to 3D
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    x_cam = (x_px - cx) * depth / fx
    y_cam = (y_px - cy) * depth / fy
    z_cam = depth
    
    return np.array([x_cam, y_cam, z_cam])


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
    """Navigate quadrotor to target waypoint"""
    
    dt = 0.01
    tolerance = 0.1
    max_time = 30.0
    
    controller = QuadrotorController(tello)
    param = tello
    
    pos = np.array(currentPose['position'])
    rpy = np.array(currentPose['rpy'])
    
    vel = np.zeros(3)
    pqr = np.zeros(3)
    
    roll, pitch, yaw = rpy
    quat = Quaternion(axis=[0, 0, 1], radians=yaw) * \
           Quaternion(axis=[0, 1, 0], radians=pitch) * \
           Quaternion(axis=[1, 0, 0], radians=roll)
    
    current_state = np.concatenate([pos, vel, [quat.x, quat.y, quat.z, quat.w], pqr])
    target_position = np.array(targetPose)
    
    distance = np.linalg.norm(target_position - pos)
    estimated_time = min(distance / velocity * 2.0, max_time)
    
    print(f"  Navigating: {pos} â†’ {target_position}")
    print(f"  Distance: {distance:.2f}m, Est. time: {estimated_time:.1f}s")
    
    if distance < tolerance:
        print("  Already at target!")
        return {'position': pos, 'rpy': rpy}
    
    # Generate trajectory (same as before)
    num_points = int(estimated_time / dt)
    time_points = np.linspace(0, estimated_time, num_points)
    
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
            vel_mag = (cruise_vel / accel_time) * t
            acc_mag = cruise_vel / accel_time
            progress = 0.5 * (cruise_vel / accel_time) * t * t / distance
        elif t <= accel_time + cruise_time:
            vel_mag = cruise_vel
            acc_mag = 0.0
            progress = (0.5 * cruise_vel * accel_time + cruise_vel * (t - accel_time)) / distance
        else:
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
    
    controller.set_trajectory(trajectory_points, time_points, velocities, accelerations)
    
    state = current_state.copy()
    
    for i, t in enumerate(time_points):
        control_input = controller.compute_control(state, t)
        
        current_pos = state[0:3]
        error = np.linalg.norm(current_pos - target_position)
        if error < tolerance and t > 1.0:
            print(f"  âœ“ Reached at t={t:.2f}s, error={error:.3f}m")
            state_final = state
            break
        
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
        state_final = state
        error = np.linalg.norm(state_final[0:3] - target_position)
        print(f"  Final error: {error:.3f}m")
    
    final_pos = state_final[0:3]
    final_quat = Quaternion(state_final[9], state_final[6], state_final[7], state_final[8])
    final_ypr = final_quat.yaw_pitch_roll
    final_rpy = np.array([final_ypr[2], final_ypr[1], final_ypr[0]])
    
    return {'position': final_pos, 'rpy': final_rpy}


################################################
#### Main Function ##############################
################################################
def main(renderer):
    init_log_dir()
    
    print("="*70)
    print("Window Navigation with TSÂ²P Detection")
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
        
        print(f"Camera: {render_resolution}x{render_resolution}, FOV: {np.degrees(fov_radians):.1f}Â°")
        
        # Initial pose - Gaussian splat coordinates
        currentPose = {
            'position': np.array([0.2, -0.2, 0.0]),  # Working start position
            'rpy': np.radians([0.0, 0.0, 0.0])
        }
        
        print(f"Initial position: {currentPose['position']}")
        
        #####################################################
        ### STEP 1: ACTIVE SCANNING
        #####################################################
        
        print("\n--- Phase 1: Active Scanning ---")
        scan_positions = detector.generate_scanning_positions(currentPose['position'])
        print(f"Generated {len(scan_positions)} scan positions")
        
        scan_frames = []
        for i, pos in enumerate(scan_positions):
            print(f"  Scan frame {i+1}/{len(scan_positions)}")
            
            color_image, depth_image, metric_depth = renderer.render(pos, currentPose['rpy'])
            scan_frames.append(color_image)
            
            cv2.imwrite(f'./log/scan_{i:02d}.png', cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
        
        print(f"  âœ“ Captured {len(scan_frames)} frames")
        
        #####################################################
        ### STEP 2: WINDOW DETECTION
        #####################################################

        print("\n--- Phase 2: TS²P Detection ---")

        # OLD methods (commented out for reference)
        # window_mask, window_center_2d, confidence = detector.detect_window(scan_frames)
        # window_mask, window_center_2d, confidence = detector.detect_window_with_flow_debug(scan_frames)

        # NEW: Use improved detector with bounding box selection
        from improved_window_detector import ImprovedWindowDetector
        improved_detector = ImprovedWindowDetector(detector)  # Wrap existing detector

        # Run improved detection
        window_mask, window_center_2d, confidence, debug_info = improved_detector.detect_window_improved(scan_frames)

        # Visualize the detection process (creates ./log/detection_process.png)
        improved_detector.visualize_detection_process(debug_info, './log/detection_process.png')

        print(f"  Confidence: {confidence:.3f}")

        if window_center_2d is None or confidence < 0.2:
            print("  ✗ Detection failed!")
            return False

        print(f"  ✓ Window at: {window_center_2d}")

        # Also create the standard detection visualization (creates ./log/detection.png)
        detector.visualize_detection(scan_frames[0], window_mask, window_center_2d,
                                    './log/detection.png')
        
        #####################################################
        ### STEP 3: 3D POSITION
        #####################################################
        
        print("\n--- Phase 3: 3D Localization ---")
        
        color_image, depth_image, metric_depth = renderer.render(
            currentPose['position'], currentPose['rpy'])
        
        window_pos_cam = compute_window_3d_position(window_center_2d, metric_depth, camera_matrix)
        window_pos_world = currentPose['position'] + window_pos_cam
        
        print(f"  Window 3D: {window_pos_world}")
        
        #####################################################
        ### STEP 4: NAVIGATE
        #####################################################
        
        print("\n--- Phase 4: Navigation ---")
        
        # Approach
        direction = window_pos_world - currentPose['position']
        direction_norm = direction / np.linalg.norm(direction)
        
        approach_target = window_pos_world - 1.0 * direction_norm
        currentPose = goToWaypoint(currentPose, approach_target, velocity=1.0)
        
        # Pass through
        final_target = window_pos_world + 2.0 * direction_norm
        currentPose = goToWaypoint(currentPose, final_target, velocity=0.8)
        
        print(f"\nâœ“ Mission Complete!")
        print(f"  Final position: {currentPose['position']}")
        
        return True
        
    except Exception as e:
        print(f"\nâœ— Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    config_path = "../data/p4_colmap_nov6_1000_splat/p4_colmap_nov6_1000/splatfacto/2025-11-06_161816/config.yml"
    json_path = "../render_settings/render_settings.json"
    
    renderer = SplatRenderer(config_path, json_path)
    success = main(renderer)
    
    if success:
        print("\nâœ“ Window cleared successfully!")
    else:
        print("\nâœ— Navigation failed")