"""
Triangulation-based depth estimation from scanning motion
Uses the same scanning sequence as TS²P detection to estimate depth via parallax
"""

import numpy as np
import cv2


class TriangulationDepthEstimator:
    """
    Estimate depth using triangulation from scanning motion
    
    Principle:
    - During scanning, camera moves by known translation (scan_distance)
    - Pixel displacement (parallax) is inversely proportional to depth
    - depth = baseline * focal_length / disparity
    """
    
    def __init__(self, camera_matrix, scan_distance):
        """
        Args:
            camera_matrix: (3, 3) camera intrinsic matrix
            scan_distance: float, total scanning distance in splat units
        """
        self.camera_matrix = camera_matrix
        self.fx = camera_matrix[0, 0]
        self.fy = camera_matrix[1, 1]
        self.cx = camera_matrix[0, 2]
        self.cy = camera_matrix[1, 2]
        self.scan_distance = scan_distance
        
        print(f"Triangulation Depth Estimator initialized:")
        print(f"  Focal length: fx={self.fx:.1f}, fy={self.fy:.1f}")
        print(f"  Principal point: cx={self.cx:.1f}, cy={self.cy:.1f}")
        print(f"  Scan distance: {self.scan_distance:.4f} splat units")
    
    def estimate_depth_from_flow(self, flow_magnitude, baseline_3d):
        """
        Estimate depth from optical flow magnitude
        
        Args:
            flow_magnitude: (H, W) numpy array of flow magnitudes in pixels
            baseline_3d: float, 3D distance traveled during scan (in splat units)
        
        Returns:
            depth_map: (H, W) numpy array of estimated depths in splat units
        """
        # Avoid division by zero
        flow_magnitude = np.maximum(flow_magnitude, 1e-6)
        
        # Depth from parallax: Z = (baseline * f) / disparity
        # Here disparity ≈ flow_magnitude (pixel displacement)
        # Using average focal length
        f_avg = (self.fx + self.fy) / 2.0
        depth_map = (baseline_3d * f_avg) / flow_magnitude
        
        # Improved sanity bounds based on typical scene
        # Windows are typically 0.2-1.0 splat units away
        # Background walls 1.0-3.0 splat units
        depth_map = np.clip(depth_map, 0.1, 3.0)
        
        return depth_map
    
    def estimate_depth_at_point(self, center_2d, flow_magnitude_map, baseline_3d, window_size=20):
        """
        Estimate depth at a specific 2D point using median flow in neighborhood
        
        Args:
            center_2d: (x, y) tuple in pixels
            flow_magnitude_map: (H, W) flow magnitude from TS²P
            baseline_3d: 3D scanning baseline
            window_size: neighborhood size for median filtering
        
        Returns:
            depth: float, estimated depth in splat units
        """
        x_px, y_px = center_2d
        H, W = flow_magnitude_map.shape
        
        # Extract local window
        y_min = max(0, y_px - window_size)
        y_max = min(H, y_px + window_size)
        x_min = max(0, x_px - window_size)
        x_max = min(W, x_px + window_size)
        
        flow_window = flow_magnitude_map[y_min:y_max, x_min:x_max]
        
        # Use median flow (robust to outliers)
        median_flow = np.median(flow_window)
        
        # IMPORTANT: Low flow at window center means it's CLOSER (low parallax)
        # This is opposite of what you'd expect!
        # Use a minimum flow threshold to avoid depth explosion
        median_flow = max(median_flow, 5.0)  # At least 5 pixels of flow
        
        # Estimate depth
        f_avg = (self.fx + self.fy) / 2.0
        depth = (baseline_3d * f_avg) / median_flow
        
        # Sanity check - windows typically 0.2-1.0 splat units
        depth = np.clip(depth, 0.2, 2.0)
        
        return depth
    
    def compute_3d_position(self, center_2d, depth):
        """
        Unproject 2D pixel + depth to 3D position in camera frame
        
        Args:
            center_2d: (x, y) tuple in pixels
            depth: float, depth in splat units
        
        Returns:
            position_3d: (3,) array [x, y, z] in camera frame (splat units)
        """
        x_px, y_px = center_2d
        
        # Unproject using pinhole camera model
        x_cam = (x_px - self.cx) * depth / self.fx
        y_cam = (y_px - self.cy) * depth / self.fy
        z_cam = depth
        
        return np.array([x_cam, y_cam, z_cam])
    
    def compute_baseline_3d(self, scan_positions):
        """
        Compute 3D baseline from scanning positions
        
        Args:
            scan_positions: list of (3,) position arrays
        
        Returns:
            baseline: float, total 3D distance traveled
        """
        start_pos = scan_positions[0]
        end_pos = scan_positions[-1]
        baseline = np.linalg.norm(end_pos - start_pos)
        return baseline


def compute_window_3d_position_triangulation(center_2d, flow_magnitude_map, 
                                             camera_matrix, scan_positions):
    """
    Compute 3D position of window center using triangulation
    
    Args:
        center_2d: (x, y) tuple, window center in pixels
        flow_magnitude_map: (H, W) flow magnitude from TS²P detection
        camera_matrix: (3, 3) camera intrinsics
        scan_positions: list of (3,) scan position arrays
    
    Returns:
        position_3d: (3,) array, window position in camera frame
    """
    # Create estimator
    scan_distance = np.linalg.norm(scan_positions[-1] - scan_positions[0])
    estimator = TriangulationDepthEstimator(camera_matrix, scan_distance)
    
    # Compute 3D baseline
    baseline_3d = estimator.compute_baseline_3d(scan_positions)
    
    print(f"  Triangulation:")
    print(f"    3D baseline: {baseline_3d:.4f} splat units")
    
    # Estimate depth at window center
    depth = estimator.estimate_depth_at_point(center_2d, flow_magnitude_map, 
                                              baseline_3d, window_size=20)
    
    print(f"    Estimated depth: {depth:.4f} splat units")
    
    # Get flow at center for validation
    x_px, y_px = center_2d
    flow_at_center = flow_magnitude_map[y_px, x_px]
    print(f"    Flow at center: {flow_at_center:.2f} pixels")
    
    # Unproject to 3D
    position_3d = estimator.compute_3d_position(center_2d, depth)
    
    print(f"    3D position (camera frame): [{position_3d[0]:.4f}, {position_3d[1]:.4f}, {position_3d[2]:.4f}]")
    
    return position_3d


def visualize_depth_map(flow_magnitude_map, camera_matrix, baseline_3d, 
                        save_path='./log/depth_from_flow.png'):
    """
    Visualize estimated depth map from flow
    
    Args:
        flow_magnitude_map: (H, W) flow magnitude
        camera_matrix: (3, 3) camera intrinsics
        baseline_3d: scanning baseline in 3D
        save_path: where to save visualization
    """
    import matplotlib.pyplot as plt
    
    # Create estimator
    estimator = TriangulationDepthEstimator(camera_matrix, baseline_3d)
    
    # Estimate depth map
    depth_map = estimator.estimate_depth_from_flow(flow_magnitude_map, baseline_3d)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Flow magnitude
    im1 = axes[0].imshow(flow_magnitude_map, cmap='jet')
    axes[0].set_title('Flow Magnitude (pixels)', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)
    
    # Estimated depth
    im2 = axes[1].imshow(depth_map, cmap='plasma', vmin=0, vmax=2.0)
    axes[1].set_title('Estimated Depth (splat units)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)
    
    # Inverse depth (disparity-like)
    inv_depth = 1.0 / (depth_map + 1e-6)
    im3 = axes[2].imshow(inv_depth, cmap='viridis')
    axes[2].set_title('Inverse Depth (1/Z)', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046)
    
    plt.suptitle('Depth Estimation from Optical Flow Triangulation', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved depth visualization to {save_path}")
    plt.close()