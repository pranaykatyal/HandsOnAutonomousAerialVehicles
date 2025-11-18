"""
window_navigation.py
Integrates TS²P gap detection with navigation stack
"""

import torch
import numpy as np
from ts2p_network import OpticalFlowExtractor, TS2P_GapDetector
from control import QuadrotorController
from trajectory_generator import TrajectoryGenerator
import cv2


class WindowNavigationSystem:
    """
    Complete navigation system for flying through unknown windows
    
    Pipeline:
    1. Active scanning (diagonal motion)
    2. TS²P gap detection
    3. Visual servoing to window center
    4. Fly through
    """
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize TS²P detector
        print("Initializing TS²P Gap Detector...")
        self.flow_extractor = OpticalFlowExtractor(model_type='raft', device=self.device)
        self.gap_detector = TS2P_GapDetector(self.flow_extractor, device=self.device)
        self.gap_detector.threshold_percentile = 83  # Optimized threshold
        
        # Initialize controller
        self.controller = QuadrotorController()
        
        # Scanning parameters (from your Blender generation)
        self.scan_distance = 2.0  # meters
        self.num_scan_frames = 5
        self.camera_start_distance = 11.0  # meters from window
        
        # Detection state
        self.window_detected = False
        self.window_center = None
        self.window_mask = None
        
        print("✓ Navigation system initialized")
    
    def detect_window_from_frames(self, frames):
        """
        Detect window from scanning sequence
        
        Args:
            frames: (N, H, W, 3) numpy array - N RGB frames
        Returns:
            mask: (H, W) binary mask
            center: (x, y) window center in pixels
            confidence: detection confidence [0, 1]
        """
        # Convert to torch tensor
        frames_tensor = torch.from_numpy(frames).float() / 255.0  # (N, H, W, 3)
        frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # (N, 3, H, W)
        frames_tensor = frames_tensor.unsqueeze(0)  # (1, N, 3, H, W)
        frames_tensor = frames_tensor.to(self.device)
        
        # Detect gap using TS²P
        with torch.no_grad():
            mask, flow_magnitude = self.gap_detector.detect_gap(frames_tensor, refine=True)
        
        # Convert to numpy
        mask = mask[0, 0].cpu().numpy()  # (H, W)
        
        # Compute window center (centroid of mask)
        if mask.sum() > 0:
            y_coords, x_coords = np.where(mask > 0.5)
            center_x = int(x_coords.mean())
            center_y = int(y_coords.mean())
            center = (center_x, center_y)
            
            # Confidence based on mask size and compactness
            mask_area = mask.sum()
            total_area = mask.shape[0] * mask.shape[1]
            confidence = min(1.0, mask_area / (total_area * 0.3))  # Normalize
        else:
            center = None
            confidence = 0.0
        
        return mask, center, confidence
    
    def compute_window_approach_waypoint(self, center, current_position, image_size):
        """
        Compute next waypoint to approach window center
        
        Args:
            center: (x, y) window center in pixels
            current_position: (x, y, z) current quadrotor position
            image_size: (H, W) image dimensions
        Returns:
            waypoint: (x, y, z) target position
        """
        H, W = image_size
        
        # Image center (where we want window center to be)
        img_center_x = W / 2
        img_center_y = H / 2
        
        # Error in pixels
        error_x = center[0] - img_center_x
        error_y = center[1] - img_center_y
        
        # Convert pixel error to world coordinates (simple proportional control)
        # Assuming camera FOV and distance
        fov_horizontal = 90  # degrees (adjust for your camera)
        fov_vertical = 60    # degrees
        
        distance_to_window = current_position[2] - self.camera_start_distance  # Depth
        
        # Pixel to meter conversion
        meters_per_pixel_x = 2 * distance_to_window * np.tan(np.radians(fov_horizontal/2)) / W
        meters_per_pixel_y = 2 * distance_to_window * np.tan(np.radians(fov_vertical/2)) / H
        
        # Target offset in world frame
        offset_x = error_x * meters_per_pixel_x
        offset_y = -error_y * meters_per_pixel_y  # Image Y is inverted
        
        # Compute waypoint (move to align window center)
        waypoint = (
            current_position[0] + offset_x * 0.3,  # Proportional gain
            current_position[1] + offset_y * 0.3,
            current_position[2] - 0.5  # Move forward slightly
        )
        
        return waypoint
    
    def visualize_detection(self, frame, mask, center, save_path=None):
        """
        Visualize detection on frame
        
        Args:
            frame: (H, W, 3) RGB image
            mask: (H, W) binary mask
            center: (x, y) or None
        """
        vis = frame.copy()
        
        # Overlay mask (green)
        mask_overlay = np.zeros_like(vis)
        mask_overlay[mask > 0.5] = [0, 255, 0]
        vis = cv2.addWeighted(vis, 0.7, mask_overlay, 0.3, 0)
        
        # Draw center
        if center is not None:
            cv2.circle(vis, center, 10, (0, 0, 255), -1)
            cv2.circle(vis, center, 15, (0, 0, 255), 2)
            
            # Draw crosshair
            H, W = frame.shape[:2]
            cv2.line(vis, (center[0], 0), (center[0], H), (0, 0, 255), 1)
            cv2.line(vis, (0, center[1]), (W, center[1]), (0, 0, 255), 1)
            
            # Draw image center (target)
            img_center = (W//2, H//2)
            cv2.circle(vis, img_center, 20, (255, 0, 0), 2)
            cv2.line(vis, (img_center[0]-25, img_center[1]), 
                     (img_center[0]+25, img_center[1]), (255, 0, 0), 2)
            cv2.line(vis, (img_center[0], img_center[1]-25), 
                     (img_center[0], img_center[1]+25), (255, 0, 0), 2)
        
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        
        return vis


# =============================================================================
# Example Usage / Test
# =============================================================================

def test_navigation_system():
    """Test the navigation system with your Blender data"""
    from gapflytdataloader import GapFlytSequenceDataset
    
    print("="*70)
    print("Testing Window Navigation System")
    print("="*70)
    
    # Load a test sequence
    dataset = GapFlytSequenceDataset(
        sequences_dir="../Blender/Outputs/Sequences",
        num_frames=5,
        random_subset=False
    )
    
    # Initialize navigation system
    nav_system = WindowNavigationSystem(device='cuda')
    
    # Test on a few sequences
    for seq_idx in [0, 50, 100]:
        print(f"\nTesting sequence {seq_idx}...")
        
        sample = dataset[seq_idx]
        frames = sample['frames'].numpy()  # (5, 3, H, W)
        frames = frames.transpose(0, 2, 3, 1)  # (5, H, W, 3)
        frames = (frames * 255).astype(np.uint8)
        
        # Detect window
        mask, center, confidence = nav_system.detect_window_from_frames(frames)
        
        print(f"  Detection confidence: {confidence:.3f}")
        if center:
            print(f"  Window center: {center}")
        else:
            print(f"  ⚠ No window detected")
        
        # Visualize
        vis = nav_system.visualize_detection(
            frames[0],  # Reference frame
            mask,
            center,
            save_path=f'nav_test_seq_{seq_idx}.png'
        )
        print(f"  ✓ Saved visualization")
    
    print("\n" + "="*70)
    print("Navigation system test complete!")
    print("="*70)


if __name__ == "__main__":
    test_navigation_system()