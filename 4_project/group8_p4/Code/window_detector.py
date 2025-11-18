"""
Window Detection using TS²P (Temporally Stacked Spatial Parallax)
Based on GapFlyt: Active Vision Based Minimalist Structure-less Gap Detection

This module implements the core TS²P algorithm for detecting windows/gaps
without requiring explicit depth information.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FlowFrame:
    """Container for storing frame and its associated optical flow"""
    image: np.ndarray
    flow: Optional[np.ndarray] = None
    timestamp: float = 0.0


class TS2PWindowDetector:
    """
    Temporally Stacked Spatial Parallax (TS²P) Window Detector
    
    Uses active vision and optical flow to detect windows without depth.
    Key principle: Foreground (closer) objects have larger optical flow
    magnitudes than background (farther) objects.
    """
    
    def __init__(self, 
                 n_frames: int = 4,
                 flow_threshold: float = 0.5,
                 min_window_area: int = 1000,
                 erosion_kernel_size: int = 5):
        """
        Initialize TS²P detector
        
        Args:
            n_frames: Number of frames to stack (N in paper)
            flow_threshold: Threshold τ' for inverse flow difference
            min_window_area: Minimum area in pixels for valid window
            erosion_kernel_size: Kernel size for morphological operations
        """
        self.n_frames = n_frames
        self.flow_threshold = flow_threshold
        self.min_window_area = min_window_area
        self.erosion_kernel_size = erosion_kernel_size  # FIX: Store this!
        self.erosion_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (erosion_kernel_size, erosion_kernel_size)
        )
        
        # Frame buffer for temporal stacking
        self.frame_buffer: List[FlowFrame] = []
        
        # Optical flow parameters (Farneback)
        self.flow_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
    def compute_optical_flow(self, 
                            img1: np.ndarray, 
                            img2: np.ndarray) -> np.ndarray:
        """
        Compute dense optical flow between two frames
        
        Args:
            img1: Reference frame (grayscale)
            img2: Current frame (grayscale)
            
        Returns:
            flow: Optical flow field (H x W x 2)
        """
        flow = cv2.calcOpticalFlowFarneback(
            img1, img2, None, **self.flow_params
        )
        return flow
    
    def compute_flow_magnitude(self, flow: np.ndarray) -> np.ndarray:
        """
        Compute magnitude of optical flow
        
        Args:
            flow: Optical flow field (H x W x 2)
            
        Returns:
            magnitude: Flow magnitude (H x W)
        """
        return np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    
    def compute_xi(self, flow_magnitudes: List[np.ndarray]) -> np.ndarray:
        """
        Compute Ξ = ∇·||ξᵢṗx||₂⁻¹
        
        This is the spatial derivative of inverse average (stacked) flow magnitudes.
        High values indicate boundary regions between foreground and background.
        
        Args:
            flow_magnitudes: List of flow magnitude images
            
        Returns:
            xi: Edge map highlighting window boundaries
        """
        # Stack and average flow magnitudes
        stacked_flow = np.stack(flow_magnitudes, axis=0)
        mean_flow = np.mean(stacked_flow, axis=0)
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-6
        inverse_mean_flow = 1.0 / (mean_flow + epsilon)
        
        # Compute spatial gradient (Sobel operators)
        sobelx = cv2.Sobel(inverse_mean_flow, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(inverse_mean_flow, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute magnitude of gradient
        xi = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize to [0, 1]
        xi = (xi - xi.min()) / (xi.max() - xi.min() + epsilon)
        
        return xi
    
    def detect_window_mask(self, xi: np.ndarray) -> np.ndarray:
        """
        Detect window mask from Ξ using distance transform approach
        
        Strategy: The window region has LOW Ξ values (smooth flow),
        while the boundary has HIGH Ξ values (flow discontinuity).
        
        Args:
            xi: Edge map from compute_xi()
            
        Returns:
            window_mask: Binary mask of window region
        """
        h, w = xi.shape
        
        # Threshold Ξ to get boundary regions (HIGH Ξ = edges)
        _, boundary_mask = cv2.threshold(
            (xi * 255).astype(np.uint8),
            int(self.flow_threshold * 255),
            255,
            cv2.THRESH_BINARY
        )
        
        # Invert: LOW Ξ regions (potential window) are white
        potential_window = cv2.bitwise_not(boundary_mask)
        
        # Use distance transform to find regions far from edges
        # These are the most "interior" regions
        dist_transform = cv2.distanceTransform(potential_window, cv2.DIST_L2, 5)
        
        # Threshold the distance transform to get seed regions
        # Regions that are at least distance 20 from any edge
        _, sure_window = cv2.threshold(dist_transform, 15, 255, cv2.THRESH_BINARY)
        sure_window = sure_window.astype(np.uint8)
        
        # Find markers for watershed
        # These are the "seeds" for different regions
        _, markers = cv2.connectedComponents(sure_window)
        
        # Use watershed to grow these seeds
        # First, we need a 3-channel image
        img_3ch = cv2.cvtColor(potential_window, cv2.COLOR_GRAY2BGR)
        
        # Watershed
        markers = markers + 1  # Background becomes 1
        
        # Mark unknown regions (neither sure bg nor sure fg) as 0
        unknown = cv2.subtract(potential_window, sure_window)
        markers[unknown == 255] = 0
        
        try:
            markers = cv2.watershed(img_3ch, markers)
        except:
            # If watershed fails, fall back to simple threshold
            pass
        
        # Extract regions
        window_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Find all unique markers (excluding -1 which is boundary and 1 which is background)
        unique_markers = np.unique(markers)
        unique_markers = unique_markers[(unique_markers > 1)]
        
        # Evaluate each region
        region_scores = []
        for marker_id in unique_markers:
            region = (markers == marker_id).astype(np.uint8) * 255
            area = np.sum(region > 0)
            
            # Check if region meets size requirement
            if area < self.min_window_area:
                continue
            
            # Get bounding box
            contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                continue
            
            cnt = contours[0]
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            
            # Check aspect ratio
            aspect_ratio = w_box / (h_box + 1e-6)
            if aspect_ratio < 0.3 or aspect_ratio > 3.0:
                continue
            
            # Check if reasonably centered (not at edges)
            margin = 20
            if x < margin or y < margin or x + w_box > w - margin or y + h_box > h - margin:
                continue
            
            # Compute centrality score (prefer centered regions)
            center_x = x + w_box / 2
            center_y = y + h_box / 2
            img_center_x = w / 2
            img_center_y = h / 2
            
            # Distance from image center (normalized)
            dist_from_center = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
            max_dist = np.sqrt((w/2)**2 + (h/2)**2)
            centrality = 1.0 - (dist_from_center / max_dist)
            
            # Combined score: larger area + more central = better
            score = area * (0.5 + 0.5 * centrality)
            
            region_scores.append((score, region))
        
        if len(region_scores) == 0:
            # Fallback: use simple largest contour approach
            return self._fallback_detection(potential_window)
        
        # Use region with best score
        region_scores.sort(key=lambda x: x[0], reverse=True)
        window_mask = region_scores[0][1]
        
        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        window_mask = cv2.morphologyEx(window_mask, cv2.MORPH_CLOSE, kernel)
        window_mask = cv2.morphologyEx(window_mask, cv2.MORPH_OPEN, kernel)
        
        # Erode for conservative estimate
        if self.erosion_kernel_size > 0:
            window_mask = cv2.erode(window_mask, self.erosion_kernel, iterations=1)
        
        return window_mask
    
    def _fallback_detection(self, potential_window: np.ndarray) -> np.ndarray:
        """
        Fallback detection method using simple contour finding
        """
        contours, _ = cv2.findContours(
            potential_window, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            return np.zeros_like(potential_window)
        
        # Find largest contour that meets criteria
        h, w = potential_window.shape
        valid_contours = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_window_area:
                continue
            
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            aspect_ratio = w_box / (h_box + 1e-6)
            
            if 0.3 < aspect_ratio < 3.0:
                valid_contours.append((area, cnt))
        
        if len(valid_contours) == 0:
            return np.zeros_like(potential_window)
        
        # Use largest valid contour
        valid_contours.sort(key=lambda x: x[0], reverse=True)
        window_mask = np.zeros_like(potential_window)
        cv2.drawContours(window_mask, [valid_contours[0][1]], 0, 255, -1)
        
        return window_mask
    
    def add_frame(self, 
                  current_frame: np.ndarray,
                  reference_frame: Optional[np.ndarray] = None) -> bool:
        """
        Add frame to buffer and compute optical flow
        
        Args:
            current_frame: Current RGB/grayscale frame
            reference_frame: Reference frame for flow computation (use first frame if None)
            
        Returns:
            ready: True if enough frames accumulated for detection
        """
        # Convert to grayscale if needed
        if len(current_frame.shape) == 3:
            gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = current_frame.copy()
        
        # Initialize reference frame
        if reference_frame is None and len(self.frame_buffer) == 0:
            self.frame_buffer.append(FlowFrame(image=gray))
            return False
        
        # Use first frame as reference if not provided
        if reference_frame is None:
            reference_frame = self.frame_buffer[0].image
        else:
            if len(reference_frame.shape) == 3:
                reference_frame = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
        
        # Compute optical flow w.r.t. reference frame
        flow = self.compute_optical_flow(reference_frame, gray)
        
        # Add to buffer
        self.frame_buffer.append(FlowFrame(image=gray, flow=flow))
        
        # Keep only last n_frames
        if len(self.frame_buffer) > self.n_frames + 1:  # +1 for reference
            self.frame_buffer.pop(1)  # Keep reference frame at index 0
        
        return len(self.frame_buffer) >= self.n_frames + 1
    
    def detect(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect window using accumulated frames
        
        Returns:
            window_mask: Binary mask of detected window (or None)
            xi: Edge map visualization (or None)
        """
        if len(self.frame_buffer) < self.n_frames + 1:
            return None, None
        
        # Extract flow magnitudes (skip reference frame)
        flow_magnitudes = []
        for frame in self.frame_buffer[1:]:
            if frame.flow is not None:
                mag = self.compute_flow_magnitude(frame.flow)
                flow_magnitudes.append(mag)
        
        if len(flow_magnitudes) < self.n_frames:
            return None, None
        
        # Compute Ξ
        xi = self.compute_xi(flow_magnitudes)
        
        # Detect window mask
        window_mask = self.detect_window_mask(xi)
        
        return window_mask, xi
    
    def reset(self):
        """Reset detector state"""
        self.frame_buffer.clear()
    
    def visualize_detection(self,
                           image: np.ndarray,
                           window_mask: np.ndarray,
                           xi: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create visualization of detection results
        
        Args:
            image: Original RGB image
            window_mask: Binary window mask
            xi: Optional edge map
            
        Returns:
            vis: Visualization image
        """
        vis = image.copy()
        
        # Overlay window mask in green
        mask_overlay = np.zeros_like(vis)
        mask_overlay[:, :, 1] = window_mask  # Green channel
        vis = cv2.addWeighted(vis, 0.7, mask_overlay, 0.3, 0)
        
        # Draw contour
        contours, _ = cv2.findContours(
            window_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
        
        return vis


def compute_safe_point(window_mask: np.ndarray) -> Optional[Tuple[int, int]]:
    """
    Compute safe point (centroid) of window for navigation
    
    Args:
        window_mask: Binary mask of window
        
    Returns:
        (cx, cy): Center point in image coordinates, or None if no valid window
    """
    # Find moments
    M = cv2.moments(window_mask)
    
    if M["m00"] == 0:
        return None
    
    # Compute centroid
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    return (cx, cy)


def estimate_window_pose_from_mask(window_mask: np.ndarray,
                                   camera_matrix: np.ndarray,
                                   assumed_window_size: float = 1.0) -> Optional[np.ndarray]:
    """
    Estimate rough 3D position of window from mask and camera parameters
    
    Args:
        window_mask: Binary window mask
        camera_matrix: 3x3 camera intrinsic matrix
        assumed_window_size: Assumed real-world window size (meters)
        
    Returns:
        position_3d: Estimated 3D position [x, y, z] in camera frame, or None
    """
    # Find window contour
    contours, _ = cv2.findContours(
        window_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    if len(contours) == 0:
        return None
    
    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Estimate depth using pinhole camera model
    # Assuming square window: real_size / image_size ≈ Z / f
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    # Use average of width and height for depth estimation
    image_size = (w + h) / 2.0
    focal_length = (fx + fy) / 2.0
    
    # Estimate depth
    Z = (assumed_window_size * focal_length) / image_size
    
    # Compute 3D position of window center
    center_x = x + w / 2.0
    center_y = y + h / 2.0
    
    X = (center_x - cx) * Z / fx
    Y = (center_y - cy) * Z / fy
    
    return np.array([X, Y, Z])


# Example usage function
def example_detection_pipeline(images: List[np.ndarray],
                               camera_matrix: np.ndarray) -> dict:
    """
    Example pipeline for window detection
    
    Args:
        images: List of images from scanning trajectory
        camera_matrix: Camera intrinsic matrix
        
    Returns:
        results: Dictionary with detection results
    """
    detector = TS2PWindowDetector(n_frames=4)
    
    # Add frames sequentially
    for i, img in enumerate(images):
        ready = detector.add_frame(img)
        
        if ready:
            print(f"Detection ready after {i+1} frames")
            break
    
    # Detect window
    window_mask, xi = detector.detect()
    
    if window_mask is None:
        return {"success": False}
    
    # Compute safe navigation point
    safe_point = compute_safe_point(window_mask)
    
    # Estimate 3D position (rough approximation)
    position_3d = estimate_window_pose_from_mask(
        window_mask, camera_matrix, assumed_window_size=1.0
    )
    
    # Create visualization
    vis = detector.visualize_detection(images[-1], window_mask, xi)
    
    return {
        "success": True,
        "window_mask": window_mask,
        "xi": xi,
        "safe_point_2d": safe_point,
        "estimated_position_3d": position_3d,
        "visualization": vis
    }