"""
main.py - Window Navigation with COORDINATE VERIFICATION
Test and verify that Y/Z movements match pixel errors correctly
"""

from splat_render import SplatRenderer
import numpy as np
import cv2
import json
from ts2p_network import OpticalFlowExtractor, TS2P_GapDetector
from scanning_fix import generate_scanning_positions_fixed
from improved_window_detector import ImprovedWindowDetector
import os
import glob


class WindowDetector:
    """Wrapper for TS²P detection"""
    
    def __init__(self, device='cuda'):
        import torch
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
    old_images = glob.glob('./log/*.png')
    if old_images:
        for img_file in old_images:
            try:
                os.remove(img_file)
            except:
                pass


def test_coordinate_mapping(renderer, camera_matrix):
    """
    Test coordinate system by moving drone and checking pixel changes
    """
    print("\n" + "="*70)
    print("COORDINATE SYSTEM TEST")
    print("="*70)
    
    base_position = np.array([0.0, 0.0, 0.0])
    rpy = np.array([0.0, 0.0, 0.0])
    
    # Render reference image
    print("\n1. Reference position (0, 0, 0)")
    img_ref, _, _ = renderer.render(base_position, rpy)
    cv2.imwrite('./log/test_ref.png', cv2.cvtColor(img_ref, cv2.COLOR_RGB2BGR))
    
    # Test Y-axis (should move window horizontally in image)
    print("\n2. Testing Y-axis (East): +0.05m")
    test_y = base_position + np.array([0.0, 0.05, 0.0])
    img_y, _, _ = renderer.render(test_y, rpy)
    cv2.imwrite('./log/test_y_plus.png', cv2.cvtColor(img_y, cv2.COLOR_RGB2BGR))
    
    print("\n3. Testing Y-axis (East): -0.05m")
    test_y_neg = base_position + np.array([0.0, -0.05, 0.0])
    img_y_neg, _, _ = renderer.render(test_y_neg, rpy)
    cv2.imwrite('./log/test_y_minus.png', cv2.cvtColor(img_y_neg, cv2.COLOR_RGB2BGR))
    
    # Test Z-axis (should move window vertically in image)
    print("\n4. Testing Z-axis (Down): +0.05m")
    test_z = base_position + np.array([0.0, 0.0, 0.05])
    img_z, _, _ = renderer.render(test_z, rpy)
    cv2.imwrite('./log/test_z_plus.png', cv2.cvtColor(img_z, cv2.COLOR_RGB2BGR))
    
    print("\n5. Testing Z-axis (Down): -0.05m")
    test_z_neg = base_position + np.array([0.0, 0.0, -0.05])
    img_z_neg, _, _ = renderer.render(test_z_neg, rpy)
    cv2.imwrite('./log/test_z_minus.png', cv2.cvtColor(img_z_neg, cv2.COLOR_RGB2BGR))
    
    print("\n✓ Test images saved to ./log/")
    print("  test_ref.png       - Reference at (0, 0, 0)")
    print("  test_y_plus.png    - Moved +Y (should show scene shifted LEFT)")
    print("  test_y_minus.png   - Moved -Y (should show scene shifted RIGHT)")
    print("  test_z_plus.png    - Moved +Z (should show scene shifted UP)")
    print("  test_z_minus.png   - Moved -Z (should show scene shifted DOWN)")
    
    print("\nKEY:")
    print("  If window RIGHT in image  → need +Y (East)")
    print("  If window LEFT in image   → need -Y (West)")
    print("  If window DOWN in image   → need +Z (Down)")
    print("  If window UP in image     → need -Z (Up)")


def main(renderer):
    init_log_dir()
    
    print("="*70)
    print("COORDINATE VERIFICATION MODE")
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
        
        focal_length = (render_resolution / 2.0) / np.tan(fov_radians / 2.0)
        cx = render_resolution / 2.0
        cy = render_resolution / 2.0
        
        camera_matrix = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ])
        
        print(f"Camera: {render_resolution}x{render_resolution}, FOV: {np.degrees(fov_radians):.1f}°")
        
        # RUN COORDINATE TEST
        test_coordinate_mapping(renderer, camera_matrix)
        
        # Initial position
        currentPos = np.array([0.0, 0.0, 0.0])
        rpy = np.array([0.0, 0.0, 0.0])
        
        print("\n" + "="*70)
        print("Phase 1: Initial Scanning & Detection")
        print("="*70)
        
        scan_positions = generate_scanning_positions_fixed(currentPos, scan_distance=0.01)
        scan_frames = []
        
        for i, pos in enumerate(scan_positions):
            img, _, _ = renderer.render(pos, rpy)
            scan_frames.append(img)
            cv2.imwrite(f'./log/scan_{i:02d}.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        improved_detector = ImprovedWindowDetector(detector)
        window_mask, window_center_2d, confidence, debug_info = improved_detector.detect_window_improved(scan_frames)
        
        if window_center_2d is None:
            print("✗ Detection failed")
            return False
        
        print(f"✓ Detected window at: {window_center_2d}")
        print(f"  Image center: ({cx:.1f}, {cy:.1f})")
        
        error_x = window_center_2d[0] - cx
        error_y = window_center_2d[1] - cy
        
        print(f"  Pixel errors:")
        print(f"    Horizontal: {error_x:+.1f}px ({'RIGHT' if error_x > 0 else 'LEFT'})")
        print(f"    Vertical:   {error_y:+.1f}px ({'DOWN' if error_y > 0 else 'UP'})")
        
        detector.visualize_detection(scan_frames[0], window_mask, window_center_2d, './log/detection.png')
        
        print("\n" + "="*70)
        print("Phase 2: Manual Correction Test")
        print("="*70)
        print("\nBased on coordinate test, apply manual correction:")
        
        # Compute correction (inverse of camera motion paradox)
        # If window is RIGHT (+error_x), camera moved LEFT, so move drone RIGHT (+Y)
        # If window is DOWN (+error_y), camera moved UP, so move drone DOWN (+Z)
        
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        
        # Use small gain for test
        gain = 0.2
        
        theta_x = np.arctan2(error_x, fx)
        theta_y = np.arctan2(error_y, fy)
        
        delta_y = gain * theta_x  # Horizontal correction
        delta_z = gain * theta_y  # Vertical correction
        
        print(f"\nComputed correction:")
        print(f"  ΔY = {delta_y:+.4f}m (horizontal)")
        print(f"  ΔZ = {delta_z:+.4f}m (vertical)")
        
        # Apply correction
        new_position = currentPos + np.array([0.0, delta_y, delta_z])
        
        print(f"\nMoving from {currentPos} to {new_position}")
        currentPos = new_position
        
        # Re-scan and detect
        print("\nRe-scanning after correction...")
        scan_positions_2 = generate_scanning_positions_fixed(currentPos, scan_distance=0.01)
        scan_frames_2 = []
        
        for pos in scan_positions_2:
            img, _, _ = renderer.render(pos, rpy)
            scan_frames_2.append(img)
        
        mask_2, center_2, conf_2 = improved_detector.detect_window_improved(scan_frames_2)[:3]
        
        if center_2 is not None:
            error_x_2 = center_2[0] - cx
            error_y_2 = center_2[1] - cy
            error_total_2 = np.sqrt(error_x_2**2 + error_y_2**2)
            
            print(f"\n✓ After correction:")
            print(f"  New pixel errors:")
            print(f"    Horizontal: {error_x_2:+.1f}px (was {error_x:+.1f}px)")
            print(f"    Vertical:   {error_y_2:+.1f}px (was {error_y:+.1f}px)")
            print(f"    Total:      {error_total_2:.1f}px (was {np.sqrt(error_x**2 + error_y**2):.1f}px)")
            
            detector.visualize_detection(scan_frames_2[0], mask_2, center_2, './log/after_correction.png')
            
            # Check if error REDUCED
            error_reduction_x = abs(error_x) - abs(error_x_2)
            error_reduction_y = abs(error_y) - abs(error_y_2)
            
            print(f"\n  Error change:")
            print(f"    Horizontal: {error_reduction_x:+.1f}px {'✓ REDUCED' if error_reduction_x > 0 else '✗ INCREASED'}")
            print(f"    Vertical:   {error_reduction_y:+.1f}px {'✓ REDUCED' if error_reduction_y > 0 else '✗ INCREASED'}")
            
            if error_reduction_x > 0 and error_reduction_y > 0:
                print("\n✓✓✓ COORDINATE SYSTEM VERIFIED! Both axes improved!")
            elif error_reduction_x < 0 or error_reduction_y < 0:
                print("\n✗✗✗ COORDINATE SIGN ERROR! Need to flip signs!")
                print("\nSuggested fix:")
                if error_reduction_x < 0:
                    print("  Flip Y-axis sign: delta_y = -gain * theta_x")
                if error_reduction_y < 0:
                    print("  Flip Z-axis sign: delta_z = -gain * theta_y")
        
        print("\n" + "="*70)
        print("Test Complete - Check log images!")
        print("="*70)
        
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
        print("\n✓ Coordinate test complete!")
    else:
        print("\n✗ Test failed")