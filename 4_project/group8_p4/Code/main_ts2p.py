# main_ts2p.py
"""
Integrated Window Detection and Navigation using TS²P Algorithm

This script integrates the TS²P (Temporally Stacked Spatial Parallax) algorithm
with the existing navigation framework.

Main workflow:
1. Execute diagonal scanning trajectory
2. Collect frames and detect window using TS²P
3. Compute safe navigation point
4. Navigate through window using goToWaypoint()
"""

import numpy as np
import cv2
import json
import os
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

# Import your existing modules (adjust paths as needed)
# from splat_render import GaussianSplatRenderer
# from control import QuadrotorController
# from quad_dynamics import QuadrotorDynamics
# from tello import TelloParams

# Import our TS²P implementation
from window_detector import TS2PWindowDetector, compute_safe_point, estimate_window_pose_from_mask
from active_scanning import ActiveScanner, compute_optimal_scan_distance


class WindowNavigationSystem:
    """
    Complete system for window detection and navigation
    """
    
    def __init__(self,
                 renderer,
                 controller,
                 dynamics,
                 camera_matrix: np.ndarray,
                 render_settings: dict):
        """
        Initialize navigation system
        
        Args:
            renderer: Gaussian splat renderer instance
            controller: Quadrotor controller instance
            dynamics: Quadrotor dynamics instance
            camera_matrix: 3x3 camera intrinsic matrix
            render_settings: Rendering configuration
        """
        self.renderer = renderer
        self.controller = controller
        self.dynamics = dynamics
        self.camera_matrix = camera_matrix
        self.render_settings = render_settings
        
        # Initialize TS²P detector
        self.detector = TS2PWindowDetector(
            n_frames=4,  # Use 4-5 frames as in paper
            flow_threshold=0.5,
            min_window_area=2000,  # Minimum 2000 pixels
            erosion_kernel_size=5
        )
        
        # Initialize scanner
        self.scanner = ActiveScanner(
            scan_distance=0.5,  # 50cm diagonal scan
            scan_direction='diagonal',
            num_waypoints=5
        )
        
        # Storage for captured frames
        self.captured_frames: List[np.ndarray] = []
        self.detection_result = None
        
    def execute_scanning_trajectory(self,
                                    start_pose: np.ndarray,
                                    capture_frequency: int = 1) -> List[np.ndarray]:
        """
        Execute scanning trajectory and capture frames
        
        Args:
            start_pose: Starting pose [x, y, z, roll, pitch, yaw]
            capture_frequency: Capture every N waypoints
            
        Returns:
            captured_frames: List of RGB images
        """
        print("Executing scanning trajectory...")
        
        # Generate waypoints
        waypoints = self.scanner.generate_scan_trajectory(
            start_pose, 
            forward_axis='z'
        )
        
        captured_frames = []
        
        # Visit each waypoint and capture frames
        for i, waypoint in enumerate(waypoints):
            # Navigate to waypoint (using your existing navigation)
            success = self.goToWaypoint(waypoint, velocity=0.5)
            
            if not success:
                print(f"Failed to reach waypoint {i}")
                continue
            
            # Capture frame if needed
            if i % capture_frequency == 0:
                # Render current view
                rgb, depth = self.renderer.render(waypoint)
                captured_frames.append(rgb)
                print(f"Captured frame {len(captured_frames)} at waypoint {i}")
        
        self.captured_frames = captured_frames
        return captured_frames
    
    def detect_window(self) -> Optional[dict]:
        """
        Detect window using TS²P algorithm on captured frames
        
        Returns:
            detection_result: Dictionary with detection results or None
        """
        if len(self.captured_frames) < 4:
            print(f"Not enough frames for detection: {len(self.captured_frames)} < 4")
            return None
        
        print(f"Running TS²P detection on {len(self.captured_frames)} frames...")
        
        # Reset detector
        self.detector.reset()
        
        # Add frames sequentially
        for i, frame in enumerate(self.captured_frames):
            ready = self.detector.add_frame(frame)
            print(f"Added frame {i+1}/{len(self.captured_frames)}, ready={ready}")
        
        # Perform detection
        window_mask, xi = self.detector.detect()
        
        if window_mask is None:
            print("Detection failed: no window mask generated")
            return None
        
        # Check if window was found
        num_pixels = np.sum(window_mask > 0)
        if num_pixels < self.detector.min_window_area:
            print(f"Detection failed: window too small ({num_pixels} pixels)")
            return None
        
        # Compute safe navigation point
        safe_point_2d = compute_safe_point(window_mask)
        
        if safe_point_2d is None:
            print("Detection failed: could not compute safe point")
            return None
        
        # Estimate 3D position
        position_3d = estimate_window_pose_from_mask(
            window_mask,
            self.camera_matrix,
            assumed_window_size=1.0  # Assume 1m x 1m window
        )
        
        # Create visualization
        vis = self.detector.visualize_detection(
            self.captured_frames[-1],
            window_mask,
            xi
        )
        
        self.detection_result = {
            "success": True,
            "window_mask": window_mask,
            "xi": xi,
            "safe_point_2d": safe_point_2d,
            "estimated_position_3d": position_3d,
            "visualization": vis,
            "num_window_pixels": num_pixels
        }
        
        print(f"✓ Window detected!")
        print(f"  - Safe point (2D): {safe_point_2d}")
        print(f"  - Estimated 3D pos: {position_3d}")
        print(f"  - Window area: {num_pixels} pixels")
        
        return self.detection_result
    
    def segmentNearestWindow(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Segment the nearest window in the image (required interface)
        
        This is called by your main.py. It returns the last detection result.
        For real-time segmentation during flight, you may need to adapt this.
        
        Args:
            rgb_image: Current RGB image
            
        Returns:
            mask: Binary mask of window (H x W)
        """
        if self.detection_result is not None:
            return self.detection_result["window_mask"]
        else:
            # Return empty mask if no detection yet
            h, w = rgb_image.shape[:2]
            return np.zeros((h, w), dtype=np.uint8)
    
    def compute_navigation_waypoint(self,
                                    current_pose: np.ndarray,
                                    offset_before_window: float = 0.5) -> Optional[np.ndarray]:
        """
        Compute waypoint for navigating through window
        
        Args:
            current_pose: Current pose [x, y, z, roll, pitch, yaw]
            offset_before_window: Stop distance before window (meters)
            
        Returns:
            waypoint: Target waypoint [x, y, z, roll, pitch, yaw] or None
        """
        if self.detection_result is None:
            return None
        
        pos_3d = self.detection_result["estimated_position_3d"]
        
        if pos_3d is None:
            return None
        
        # Convert from camera frame to world frame
        # (You'll need to adjust this based on your coordinate system)
        x_cam, y_cam, z_cam = pos_3d
        
        # Assuming camera frame: x=right, y=down, z=forward
        # World frame: x=forward, y=left, z=up (adjust as needed)
        
        # For now, simple approach: move toward window along camera z-axis
        current_x, current_y, current_z = current_pose[:3]
        roll, pitch, yaw = current_pose[3:]
        
        # Compute target in world frame (simplified - you may need proper transformation)
        # Move forward by (z_cam - offset_before_window)
        target_distance = z_cam - offset_before_window
        
        # Move along current heading direction
        target_x = current_x + target_distance * np.cos(yaw)
        target_y = current_y + target_distance * np.sin(yaw)
        target_z = current_z  # Keep same altitude
        
        waypoint = np.array([target_x, target_y, target_z, roll, pitch, yaw])
        
        return waypoint
    
    def navigate_through_window(self,
                                current_pose: np.ndarray,
                                velocity: float = 0.5) -> bool:
        """
        Navigate through detected window
        
        Args:
            current_pose: Current pose
            velocity: Navigation velocity (m/s)
            
        Returns:
            success: True if navigation successful
        """
        if self.detection_result is None:
            print("Cannot navigate: no window detected")
            return False
        
        # Compute waypoint
        waypoint = self.compute_navigation_waypoint(current_pose)
        
        if waypoint is None:
            print("Cannot navigate: failed to compute waypoint")
            return False
        
        print(f"Navigating to waypoint: {waypoint[:3]}")
        
        # Navigate using existing goToWaypoint function
        success = self.goToWaypoint(waypoint, velocity=velocity)
        
        return success
    
    def goToWaypoint(self,
                     waypoint: np.ndarray,
                     velocity: float = 0.5,
                     tolerance: float = 0.1) -> bool:
        """
        Navigate to waypoint (placeholder - use your existing implementation)
        
        Args:
            waypoint: Target pose [x, y, z, roll, pitch, yaw]
            velocity: Maximum velocity (m/s)
            tolerance: Position tolerance (meters)
            
        Returns:
            success: True if waypoint reached
        """
        # TODO: Use your actual goToWaypoint implementation from main.py
        # This is just a placeholder
        print(f"goToWaypoint called: pos={waypoint[:3]}, vel={velocity}")
        return True
    
    def save_visualization(self, output_path: str):
        """
        Save detection visualization
        
        Args:
            output_path: Path to save image
        """
        if self.detection_result is None:
            print("No detection result to visualize")
            return
        
        vis = self.detection_result["visualization"]
        cv2.imwrite(output_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"Saved visualization to {output_path}")


def main():
    """
    Main function integrating TS²P with your existing system
    """
    
    # Load configuration
    with open('render_settings/render_settings.json', 'r') as f:
        render_settings = json.load(f)
    
    # Extract camera matrix
    camera_matrix = np.array(render_settings['camera_matrix'])
    
    print("=" * 60)
    print("Window Detection and Navigation using TS²P")
    print("=" * 60)
    
    # Initialize your existing components (adjust as needed)
    # renderer = GaussianSplatRenderer(...)
    # controller = QuadrotorController(...)
    # dynamics = QuadrotorDynamics(...)
    
    # For now, placeholder
    renderer = None
    controller = None
    dynamics = None
    
    # Initialize navigation system
    nav_system = WindowNavigationSystem(
        renderer=renderer,
        controller=controller,
        dynamics=dynamics,
        camera_matrix=camera_matrix,
        render_settings=render_settings
    )
    
    # Define starting pose
    start_pose = np.array([0, 0, 1.5, 0, 0, 0])  # [x, y, z, roll, pitch, yaw]
    
    # Step 1: Execute scanning trajectory
    print("\n[1] Executing scanning trajectory...")
    frames = nav_system.execute_scanning_trajectory(start_pose)
    print(f"Captured {len(frames)} frames")
    
    # Step 2: Detect window
    print("\n[2] Detecting window using TS²P...")
    detection = nav_system.detect_window()
    
    if detection is None:
        print("✗ Window detection failed!")
        return
    
    # Step 3: Save visualization
    print("\n[3] Saving visualization...")
    nav_system.save_visualization("window_detection_result.png")
    
    # Step 4: Navigate through window
    print("\n[4] Navigating through window...")
    success = nav_system.navigate_through_window(
        current_pose=start_pose,
        velocity=0.5
    )
    
    if success:
        print("✓ Navigation successful!")
    else:
        print("✗ Navigation failed!")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()