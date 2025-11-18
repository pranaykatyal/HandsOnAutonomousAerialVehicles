"""
Active Vision Scanning Strategy for Window Detection

Implements the diagonal scanning trajectory used in GapFlyt paper
to acquire multiple viewpoints for TS²P algorithm.
"""

import numpy as np
from typing import List, Tuple


class ActiveScanner:
    """
    Generates scanning trajectories for active vision-based gap detection
    
    The quadrotor executes a controlled motion to capture multiple frames
    from different viewpoints, enabling temporal stacking of optical flow.
    """
    
    def __init__(self,
                 scan_distance: float = 0.5,
                 scan_direction: str = 'diagonal',
                 num_waypoints: int = 5):
        """
        Initialize active scanner
        
        Args:
            scan_distance: Total distance to move during scan (meters)
            scan_direction: 'diagonal', 'horizontal', or 'vertical'
            num_waypoints: Number of waypoints in scanning trajectory
        """
        self.scan_distance = scan_distance
        self.scan_direction = scan_direction
        self.num_waypoints = num_waypoints
        
    def generate_scan_trajectory(self,
                                 start_pose: np.ndarray,
                                 forward_axis: str = 'z') -> List[np.ndarray]:
        """
        Generate scanning waypoints from start pose
        
        Args:
            start_pose: Starting pose [x, y, z, roll, pitch, yaw]
            forward_axis: Which axis points forward ('x', 'y', or 'z')
            
        Returns:
            waypoints: List of waypoints [x, y, z, roll, pitch, yaw]
        """
        waypoints = []
        
        x0, y0, z0 = start_pose[:3]
        roll, pitch, yaw = start_pose[3:]
        
        # Generate waypoints based on scan direction
        for i in range(self.num_waypoints):
            t = i / (self.num_waypoints - 1)  # Normalized time [0, 1]
            offset = t * self.scan_distance
            
            if self.scan_direction == 'diagonal':
                # Diagonal scan in XZ plane (typical for GapFlyt)
                if forward_axis == 'z':
                    x = x0 - offset / np.sqrt(2)  # Move left
                    y = y0
                    z = z0 + offset / np.sqrt(2)  # Move forward
                elif forward_axis == 'x':
                    x = x0 + offset / np.sqrt(2)
                    y = y0 - offset / np.sqrt(2)
                    z = z0
                else:  # forward_axis == 'y'
                    x = x0 - offset / np.sqrt(2)
                    y = y0 + offset / np.sqrt(2)
                    z = z0
                    
            elif self.scan_direction == 'horizontal':
                # Horizontal scan
                x = x0 - offset
                y = y0
                z = z0
                
            elif self.scan_direction == 'vertical':
                # Vertical scan
                x = x0
                y = y0 + offset
                z = z0
                
            else:
                raise ValueError(f"Unknown scan direction: {self.scan_direction}")
            
            # Keep orientation constant during scan
            waypoint = np.array([x, y, z, roll, pitch, yaw])
            waypoints.append(waypoint)
        
        return waypoints
    
    def generate_approach_trajectory(self,
                                     current_pose: np.ndarray,
                                     target_position: np.ndarray,
                                     approach_distance: float = 0.3,
                                     num_waypoints: int = 10) -> List[np.ndarray]:
        """
        Generate smooth approach trajectory toward window
        
        Args:
            current_pose: Current pose [x, y, z, roll, pitch, yaw]
            target_position: Target 3D position [x, y, z]
            approach_distance: Stop distance before window (meters)
            num_waypoints: Number of waypoints in trajectory
            
        Returns:
            waypoints: List of waypoints for approach
        """
        waypoints = []
        
        x0, y0, z0 = current_pose[:3]
        roll, pitch, yaw = current_pose[3:]
        
        xt, yt, zt = target_position
        
        # Compute direction vector
        direction = np.array([xt - x0, yt - y0, zt - z0])
        distance = np.linalg.norm(direction)
        
        if distance < approach_distance:
            return [current_pose]  # Already at target
        
        # Normalize direction
        direction = direction / distance
        
        # Target point is approach_distance before window
        target_distance = distance - approach_distance
        
        # Generate waypoints along direction
        for i in range(num_waypoints):
            t = i / (num_waypoints - 1)
            d = t * target_distance
            
            pos = np.array([x0, y0, z0]) + d * direction
            
            # Compute yaw to face target
            dx, dy = direction[0], direction[1]
            target_yaw = np.arctan2(dy, dx)
            
            waypoint = np.array([
                pos[0], pos[1], pos[2],
                roll, pitch, target_yaw
            ])
            waypoints.append(waypoint)
        
        return waypoints


class AdaptiveScanner(ActiveScanner):
    """
    Adaptive scanner that adjusts trajectory based on detection confidence
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.detection_quality_threshold = 0.7
        
    def needs_rescan(self, 
                     window_mask: np.ndarray,
                     xi_quality: float) -> bool:
        """
        Determine if rescanning is needed for better detection
        
        Args:
            window_mask: Detected window mask
            xi_quality: Quality metric of Ξ (0-1)
            
        Returns:
            needs_rescan: True if detection quality is low
        """
        if window_mask is None:
            return True
        
        # Check if mask is too small or fragmented
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            window_mask.astype(np.uint8)
        )
        
        # Fragmented detection (multiple components)
        if num_labels > 2:
            return True
        
        # Low quality Ξ
        if xi_quality < self.detection_quality_threshold:
            return True
        
        return False
    
    def adjust_scan_parameters(self,
                               detection_failed: bool,
                               current_scan_distance: float) -> float:
        """
        Adjust scanning parameters for next attempt
        
        Args:
            detection_failed: Whether previous detection failed
            current_scan_distance: Current scan distance
            
        Returns:
            new_scan_distance: Adjusted scan distance
        """
        if detection_failed:
            # Increase scan distance for more motion parallax
            return min(current_scan_distance * 1.5, 1.0)
        else:
            return current_scan_distance


def compute_optimal_scan_distance(estimated_depth: float,
                                   baseline_ratio: float = 0.1) -> float:
    """
    Compute optimal scanning distance based on estimated scene depth
    
    Args:
        estimated_depth: Estimated distance to window (meters)
        baseline_ratio: Ratio of baseline to depth (0.1 = 10%)
        
    Returns:
        scan_distance: Recommended scanning distance
    """
    # Larger baseline for farther objects
    scan_distance = estimated_depth * baseline_ratio
    
    # Clamp to reasonable range
    scan_distance = np.clip(scan_distance, 0.2, 1.0)
    
    return scan_distance


# Example usage
if __name__ == "__main__":
    # Create scanner
    scanner = ActiveScanner(
        scan_distance=0.5,
        scan_direction='diagonal',
        num_waypoints=5
    )
    
    # Generate scanning trajectory
    start_pose = np.array([0, 0, 0, 0, 0, 0])  # [x, y, z, roll, pitch, yaw]
    waypoints = scanner.generate_scan_trajectory(start_pose, forward_axis='z')
    
    print("Scanning waypoints:")
    for i, wp in enumerate(waypoints):
        print(f"  WP{i}: pos=[{wp[0]:.3f}, {wp[1]:.3f}, {wp[2]:.3f}], "
              f"orient=[{wp[3]:.3f}, {wp[4]:.3f}, {wp[5]:.3f}]")
    
    # Generate approach trajectory
    target_pos = np.array([2.0, 0.5, 3.0])
    approach_wps = scanner.generate_approach_trajectory(
        start_pose, target_pos, approach_distance=0.3
    )
    
    print(f"\nApproach trajectory ({len(approach_wps)} waypoints)")