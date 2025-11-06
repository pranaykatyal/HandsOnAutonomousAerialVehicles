import numpy as np
import cv2


class WindowDetector:
    """
    Detects windows from segmentation masks and estimates their pose
    Uses pure monocular visual servoing (no depth sensor needed for bonus!)
    """
    
    def __init__(self, image_width=1440, image_height=1080):
        """
        Initialize detector with image dimensions
        
        Parameters:
        - image_width: Image width in pixels (from render settings)
        - image_height: Image height in pixels
        """
        self.img_width = image_width
        self.img_height = image_height
        self.img_center = np.array([image_width / 2, image_height / 2])
        self.image_area = image_width * image_height
        
        # Tuning parameters
        self.min_area_threshold = 5000  # Minimum area in pixels to be considered a window
        self.alignment_threshold = 30  # Pixels - how aligned must we be before flying through
        self.close_area_ratio = 0.50  # When window fills 35% of image, it's close enough
        
    def process_segmentation(self, seg_mask):
        """
        Extract all window detections from binary segmentation mask
        
        Parameters:
        - seg_mask: Binary mask (H, W) - uint8 with 255 for window, 0 for background
        
        Returns: list of detection dictionaries with keys:
            - 'contour': OpenCV contour
            - 'bbox': (x, y, w, h) bounding box
            - 'area': area in pixels
            - 'center': (cx, cy) center point in pixels
            - 'aspect_ratio': width/height ratio
            - 'corners': (4, 2) array of corner points [TL, TR, BR, BL]
        """
        # Ensure mask is binary
        if seg_mask.dtype != np.uint8:
            seg_mask = (seg_mask * 255).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter noise
            if area < self.min_area_threshold:
                continue
            
            # Bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Center of mass (more robust than bbox center)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w//2, y + h//2
            
            # Get oriented bounding box for corners
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # Order corners consistently: TL, TR, BR, BL
            corners = self._order_corners(box)
            
            detection = {
                'contour': contour,
                'bbox': (x, y, w, h),
                'area': area,
                'center': np.array([cx, cy]),
                'aspect_ratio': w / h if h > 0 else 1.0,
                'corners': corners
            }
            
            detections.append(detection)
        
        return detections
    
    def _order_corners(self, pts):
        """
        Order corners consistently: top-left, top-right, bottom-right, bottom-left
        
        Parameters:
        - pts: (4, 2) array of corner points
        
        Returns: (4, 2) array ordered as [TL, TR, BR, BL]
        """
        # Sort by y-coordinate
        pts = pts[pts[:, 1].argsort()]
        
        # Top two points
        top = pts[:2]
        top = top[top[:, 0].argsort()]  # Sort by x: left, right
        
        # Bottom two points
        bottom = pts[2:]
        bottom = bottom[bottom[:, 0].argsort()]  # Sort by x: left, right
        
        return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)
    
    def get_closest_window(self, detections):
        """
        Find the closest window (largest area = closest due to perspective)
        
        Parameters:
        - detections: list of detection dictionaries
        
        Returns: single detection dictionary or None
        """
        if not detections:
            return None
        
        # Largest window area = closest window
        closest = max(detections, key=lambda d: d['area'])
        
        return closest
    
    def calculate_alignment_error(self, window_detection):
        """
        Calculate how misaligned the window is from image center
        
        Parameters:
        - window_detection: detection dictionary
        
        Returns: tuple of (error_x, error_y, error_magnitude) in pixels
            - error_x: horizontal error (negative = window left of center)
            - error_y: vertical error (negative = window above center)
            - error_magnitude: Euclidean distance in pixels
        """
        window_center = window_detection['center']
        
        error_x = window_center[0] - self.img_center[0]
        error_y = window_center[1] - self.img_center[1]
        error_magnitude = np.linalg.norm([error_x, error_y])
        
        return error_x, error_y, error_magnitude
    
    def is_aligned(self, window_detection):
        """Check if window is aligned with image center"""
        _, _, error_mag = self.calculate_alignment_error(window_detection)
        return error_mag < self.alignment_threshold
    
    def is_close_enough(self, window_detection):
        """
        Check if window is close enough to fly through
        Uses area ratio as proxy for distance (no depth needed!)
        """
        area_ratio = window_detection['area'] / self.image_area
        return area_ratio > self.close_area_ratio
    
    def get_approach_distance(self, window_detection):
        """
        Estimate how far to move forward (in normalized units)
        Based on window size - larger window = closer = less distance to go
        
        Returns: distance factor (0.0 = at window, 1.0 = far away)
        """
        area_ratio = window_detection['area'] / self.image_area
        
        # Simple inverse relationship
        # When area_ratio = 0.35 (close_area_ratio), distance = 0
        # When area_ratio = 0.05 (small/far), distance = 1.0
        
        if area_ratio >= self.close_area_ratio:
            return 0.0  # Already at window
        
        # Linear interpolation
        min_ratio = 0.02  # Very far away
        distance_factor = 1.0 - (area_ratio - min_ratio) / (self.close_area_ratio - min_ratio)
        distance_factor = np.clip(distance_factor, 0.0, 1.0)
        
        return distance_factor
    
    def compute_navigation_target(self, current_position, window_detection, forward_distance=1.0):
        """
        Compute target waypoint to navigate towards window center
        
        Parameters:
        - current_position: [x, y, z] current position in NED frame
        - window_detection: detection dictionary
        - forward_distance: how far forward to move (meters)
        
        Returns: [x, y, z] target position in NED frame
        """
        error_x, error_y, _ = self.calculate_alignment_error(window_detection)
        
        # Convert pixel error to lateral movement (rough estimate)
        # Assuming ~1 pixel = ~0.001 meters at current distance
        # This is a rough heuristic - adjust based on camera FOV and distance
        
        # Scale pixel error to meters (you may need to tune these)
        scale_x = 0.0005  # meters per pixel
        scale_y = 0.0005  # meters per pixel
        
        lateral_x = -error_y * scale_y  # Image Y -> NED X (left/right)
        lateral_y = error_x * scale_x   # Image X -> NED Y (forward/back in horizontal)
        
        # ADD: Limit lateral corrections
        max_lateral = 0.3  # meters
        lateral_x = np.clip(lateral_x, -max_lateral, max_lateral)
        lateral_y = np.clip(lateral_y, -max_lateral, max_lateral)
        
        # Compute target
        target = current_position.copy()
        target[0] += forward_distance  # Forward in NED frame (North)
        target[1] += lateral_y         # Lateral correction
        target[2] += lateral_x         # Vertical correction (if needed, usually keep constant)
        
        return target
    
    def visualize_detection(self, image, detections, closest_window=None):
        """
        Draw detection visualization on image
        
        Parameters:
        - image: RGB image (H, W, 3)
        - detections: list of all detections
        - closest_window: highlighted closest window detection
        
        Returns: annotated image
        """
        vis_img = image.copy()
        
        # Draw all detections in green
        for det in detections:
            cv2.drawContours(vis_img, [det['contour']], -1, (0, 255, 0), 2)
            cx, cy = det['center']
            cv2.circle(vis_img, (cx, cy), 5, (0, 255, 0), -1)
        
        # Highlight closest window in red
        if closest_window is not None:
            cv2.drawContours(vis_img, [closest_window['contour']], -1, (0, 0, 255), 3)
            cx, cy = closest_window['center']
            cv2.circle(vis_img, (cx, cy), 8, (0, 0, 255), -1)
            
            # Draw corners
            corners = closest_window['corners']
            for corner in corners:
                cv2.circle(vis_img, tuple(corner.astype(int)), 5, (255, 0, 0), -1)
            
            # Draw bounding box
            x, y, w, h = closest_window['bbox']
            cv2.rectangle(vis_img, (x, y), (x+w, y+h), (255, 255, 0), 2)
            
            # Draw crosshair at image center
            img_cx, img_cy = self.img_center.astype(int)
            cv2.drawMarker(vis_img, (img_cx, img_cy), (255, 0, 255), 
                          cv2.MARKER_CROSS, 30, 2)
            
            # Draw line from center to window center
            cv2.line(vis_img, (img_cx, img_cy), (cx, cy), (255, 0, 255), 2)
            
            # Add text info
            area_pct = closest_window['area'] / self.image_area * 100
            error_x, error_y, error_mag = self.calculate_alignment_error(closest_window)
            
            text = f"Area: {area_pct:.1f}% | Error: {error_mag:.0f}px"
            cv2.putText(vis_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
            
            status = "ALIGNED & CLOSE" if (self.is_aligned(closest_window) and 
                                          self.is_close_enough(closest_window)) else \
                    "ALIGNED" if self.is_aligned(closest_window) else \
                    "APPROACHING"
            cv2.putText(vis_img, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0) if "CLOSE" in status else (255, 255, 0), 2)
        
        return vis_img