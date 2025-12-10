"""
Window Pose Estimation using OpenCV's PnP (Perspective-n-Point)
MINIMAL FIX: Just downsample the image to match mask resolution
No changes needed to other files!
"""

import numpy as np
import cv2


class WindowPnPEstimator:
    """
    Estimate window pose using PnP from detected mask
    FIXED: Downsamples image to match mask resolution
    """
    
    def __init__(self, camera_matrix, dist_coeffs=None, window_width=1.0, window_height=1.0):
        """
        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: Distortion coefficients (typically None for Gaussian splatting)
            window_width: Real-world window width (meters)
            window_height: Real-world window height (meters)
        """
        self.camera_matrix_original = camera_matrix  # Store original
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros(5)
        self.window_width = window_width
        self.window_height = window_height
        
        # Define 3D window corners in window frame (centered at origin)
        w = self.window_width / 2
        h = self.window_height / 2
        
        self.object_points_3d = np.array([
            [-w, -h, 0],  # Bottom-left
            [ w, -h, 0],  # Bottom-right
            [ w,  h, 0],  # Top-right
            [-w,  h, 0],  # Top-left
        ], dtype=np.float32)
        
        print(f"✓ PnP Estimator initialized")
        print(f"  Window size: {window_width:.2f}m x {window_height:.2f}m")
        print(f"  Camera matrix:\n{camera_matrix}")
    
    def _scale_camera_matrix(self, original_width, original_height, target_width, target_height):
        """
        Scale camera matrix when image is resized
        
        Args:
            original_width, original_height: Original image dimensions
            target_width, target_height: Target image dimensions
            
        Returns:
            K_scaled: Camera matrix for target resolution
        """
        scale_x = target_width / original_width
        scale_y = target_height / original_height
        
        K_scaled = self.camera_matrix_original.copy()
        K_scaled[0, 0] *= scale_x  # fx
        K_scaled[1, 1] *= scale_y  # fy
        K_scaled[0, 2] *= scale_x  # cx
        K_scaled[1, 2] *= scale_y  # cy
        
        return K_scaled
    
    def extract_window_corners(self, mask):
        """
        Extract 4 corners from binary mask - PRESERVES TRAPEZOID SHAPE
        
        Args:
            mask: (H, W) binary mask [0, 1]
            
        Returns:
            corners_2d: (4, 2) array of corner pixel coordinates [x, y]
                       Order: [BL, BR, TR, TL]
                       Returns None if extraction fails
        """
        print(f"  extract_window_corners: mask shape = {mask.shape}, dtype = {mask.dtype}")
        print(f"    mask range: [{mask.min():.3f}, {mask.max():.3f}]")
        print(f"    nonzero pixels: {np.sum(mask > 0.5)}")
        
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"    Found {len(contours)} contours")
        
        if len(contours) == 0:
            print("  ✗ No contours in mask")
            return None
        
        # Get LARGEST contour by area
        contour_areas = [cv2.contourArea(c) for c in contours]
        print(f"    Contour areas: {contour_areas}")
        
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        
        print(f"    Using largest contour: area = {area:.0f} pixels")
        
        if area < 100:
            print("  ✗ Contour too small")
            return None
        
        # METHOD 1: Polygon approximation to get 4 corners (preserves trapezoid)
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        print(f"    Polygon approximation: {len(approx)} vertices")
        
        # If we get exactly 4 vertices, use them directly
        if len(approx) == 4:
            corners = approx.reshape(4, 2)
            print(f"    Perfect quadrilateral found!")
        else:
            # Otherwise, try increasing epsilon to simplify
            for mult in [0.03, 0.04, 0.05, 0.06]:
                epsilon = mult * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) == 4:
                    corners = approx.reshape(4, 2)
                    print(f"    Quadrilateral found with epsilon={mult}")
                    break
            else:
                # Fallback: use minAreaRect if polygon approximation fails
                print(f"    Fallback to minAreaRect")
                rect = cv2.minAreaRect(contour)
                corners = cv2.boxPoints(rect)
                corners = np.int0(corners)
        
        print(f"    Raw corners: {corners}")
        
        # Order corners: [BL, BR, TR, TL]
        corners_2d = self._order_corners(corners)
        
        print(f"    Ordered corners: {corners_2d}")
        
        return corners_2d.astype(np.float32)
    
    def _order_corners(self, pts):
        """
        Order corners consistently: [BL, BR, TR, TL]
        
        Args:
            pts: (4, 2) array of corner points (any order)
            
        Returns:
            ordered: (4, 2) array [bottom-left, bottom-right, top-right, top-left]
        """
        # Sort by y-coordinate
        pts_sorted = pts[pts[:, 1].argsort()]
        
        # Bottom two points (higher y = lower in image)
        bottom_pts = pts_sorted[2:]
        # Top two points
        top_pts = pts_sorted[:2]
        
        # Sort bottom points by x: [left, right]
        bottom_pts = bottom_pts[bottom_pts[:, 0].argsort()]
        # Sort top points by x: [left, right]
        top_pts = top_pts[top_pts[:, 0].argsort()]
        
        # Order: [BL, BR, TR, TL]
        ordered = np.array([
            bottom_pts[0],  # BL
            bottom_pts[1],  # BR
            top_pts[1],     # TR
            top_pts[0],     # TL
        ])
        
        return ordered
    
    def estimate_pose(self, mask, method=cv2.SOLVEPNP_ITERATIVE):
        """
        Estimate window pose from mask
        
        Args:
            mask: (H, W) binary mask of detected window
            method: PnP algorithm (SOLVEPNP_ITERATIVE, SOLVEPNP_EPNP, etc.)
            
        Returns:
            success: bool
            tvec: (3,) translation vector [camera frame]
            rvec: (3,) rotation vector (Rodrigues)
            corners_2d: (4, 2) detected corners for visualization
        """
        # Extract corners
        corners_2d = self.extract_window_corners(mask)
        
        if corners_2d is None:
            return False, None, None, None
        
        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(
            self.object_points_3d,
            corners_2d,
            self.camera_matrix_original,
            self.dist_coeffs,
            flags=method
        )
        
        if not success:
            print("  ✗ PnP solve failed")
            return False, None, None, corners_2d
        
        # Extract as 1D arrays
        tvec = tvec.flatten()
        rvec = rvec.flatten()
        
        return True, tvec, rvec, corners_2d
    
    def transform_to_ned(self, tvec_cam, rvec_cam, drone_pose_ned):
        """
        Transform window pose from camera frame to NED world frame
        
        Args:
            tvec_cam: (3,) translation in camera frame [X_cam, Y_cam, Z_cam]
            rvec_cam: (3,) rotation vector in camera frame
            drone_pose_ned: dict with 'position' (NED) and 'rpy' (NED Euler angles)
            
        Returns:
            window_pos_ned: (3,) window position in NED frame
            window_rpy_ned: (3,) window orientation in NED frame [roll, pitch, yaw]
        """
        # === STEP 1: Camera pose in NED ===
        drone_pos_ned = drone_pose_ned['position']
        drone_rpy_ned = drone_pose_ned['rpy']
        
        # Drone orientation as rotation matrix (NED)
        R_drone_ned = self._euler_to_rotation_matrix(drone_rpy_ned[0], drone_rpy_ned[1], drone_rpy_ned[2])
        
        # === STEP 2: Camera-to-NED coordinate system conversion ===
        # Coordinate change matrix: Camera → NED (when drone is level)
        R_cam_to_ned_level = np.array([
            [0, 0, 1],  # NED X = Camera Z (forward)
            [1, 0, 0],  # NED Y = Camera X (right)
            [0, 1, 0],  # NED Z = Camera Y (down)
        ])
        
        # === STEP 3: Window position in NED ===
        # Transform camera-frame translation to NED
        tvec_cam_as_ned = R_cam_to_ned_level @ tvec_cam
        window_pos_ned = drone_pos_ned + R_drone_ned @ tvec_cam_as_ned
        
        # === STEP 4: Window orientation in NED ===
        # Convert rotation vector to matrix
        R_window_cam, _ = cv2.Rodrigues(rvec_cam)
        
        # Transform to NED
        R_window_ned = R_drone_ned @ R_cam_to_ned_level @ R_window_cam
        
        # Convert to Euler angles (NED convention)
        window_rpy_ned = self._rotation_matrix_to_euler(R_window_ned)
        
        return window_pos_ned, window_rpy_ned
    
    def _euler_to_rotation_matrix(self, roll, pitch, yaw):
        """Convert Euler angles (roll, pitch, yaw) to rotation matrix"""
        # ZYX convention (yaw-pitch-roll)
        cr = np.cos(roll)
        sr = np.sin(roll)
        cp = np.cos(pitch)
        sp = np.sin(pitch)
        cy = np.cos(yaw)
        sy = np.sin(yaw)
        
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr           ]
        ])
        
        return R
    
    def _rotation_matrix_to_euler(self, R):
        """Convert rotation matrix to Euler angles (roll, pitch, yaw)"""
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        
        singular = sy < 1e-6
        
        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0
        
        return np.array([roll, pitch, yaw])
    
    def visualize_pnp_result(self, image, corners_2d, tvec_cam, rvec_cam, mask=None, save_path=None):
        """
        Visualize PnP result with MASK OVERLAY and CONTOUR
        FIXED: Downsamples image to match mask resolution
        
        Args:
            image: (H, W, 3) RGB image at ORIGINAL resolution
            corners_2d: (4, 2) detected corners at MASK resolution
            tvec_cam: (3,) translation vector
            rvec_cam: (3,) rotation vector
            mask: (H_mask, W_mask) binary mask (optional, for overlay)
            save_path: Optional path to save
        """
        # CRITICAL FIX: Downsample image to match mask resolution
        if mask is not None:
            mask_h, mask_w = mask.shape[:2]
            img_h, img_w = image.shape[:2]
            
            print(f"  Visualization dimensions:")
            print(f"    Original image: {img_w}x{img_h}")
            print(f"    Mask: {mask_w}x{mask_h}")
            
            # If dimensions don't match, downsample image to mask resolution
            if (img_h, img_w) != (mask_h, mask_w):
                print(f"    → Downsampling image to {mask_w}x{mask_h}")
                vis_img = cv2.resize(image, (mask_w, mask_h), interpolation=cv2.INTER_LINEAR)
                
                # Scale camera matrix to match
                camera_matrix = self._scale_camera_matrix(img_w, img_h, mask_w, mask_h)
            else:
                vis_img = image.copy()
                camera_matrix = self.camera_matrix_original
        else:
            vis_img = image.copy()
            camera_matrix = self.camera_matrix_original
        
        # OVERLAY MASK if provided
        if mask is not None:
            mask_uint8 = (mask * 255).astype(np.uint8)
            
            # Verify dimensions NOW match
            if mask.shape[:2] != vis_img.shape[:2]:
                print(f"  ✗ ERROR: Mask shape {mask.shape[:2]} != Image shape {vis_img.shape[:2]}")
                return vis_img
            
            # Draw actual contour in cyan for comparison
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                largest_contour = max(contours, key=cv2.contourArea)
                cv2.drawContours(vis_img, [largest_contour], -1, (0, 255, 255), 2)  # Cyan contour
            
            # Semi-transparent green overlay
            mask_color = np.zeros_like(vis_img)
            mask_color[mask > 0.5] = [0, 255, 0]  # Green
            vis_img = cv2.addWeighted(vis_img, 0.7, mask_color, 0.3, 0)
        else:
            camera_matrix = self.camera_matrix_original
        
        # Draw PnP corners (magenta)
        for i, corner in enumerate(corners_2d):
            x, y = int(corner[0]), int(corner[1])
            cv2.circle(vis_img, (x, y), 8, (255, 0, 255), -1)  # Magenta
            cv2.putText(vis_img, str(i), (x+10, y+10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # Draw PnP rectangle (magenta)
        corners_int = corners_2d.astype(np.int32)
        cv2.polylines(vis_img, [corners_int], True, (255, 0, 255), 3)  # Magenta box
        
        # Draw coordinate axes
        axis_length = self.window_width * 0.5
        axis_3d = np.array([
            [0, 0, 0],
            [axis_length, 0, 0],  # X-axis (red)
            [0, axis_length, 0],  # Y-axis (green)
            [0, 0, axis_length],  # Z-axis (blue)
        ], dtype=np.float32)
        
        axis_2d, _ = cv2.projectPoints(
            axis_3d, rvec_cam, tvec_cam,
            camera_matrix, self.dist_coeffs
        )
        axis_2d = axis_2d.reshape(-1, 2).astype(np.int32)
        
        origin = tuple(axis_2d[0])
        cv2.line(vis_img, origin, tuple(axis_2d[1]), (0, 0, 255), 3)  # X - red
        cv2.line(vis_img, origin, tuple(axis_2d[2]), (0, 255, 0), 3)  # Y - green
        cv2.line(vis_img, origin, tuple(axis_2d[3]), (255, 0, 0), 3)  # Z - blue
        
        # Add legend
        distance = np.linalg.norm(tvec_cam)
        legend_y = 30
        cv2.putText(vis_img, f"Dist: {distance:.2f}m", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(vis_img, "Cyan=Contour, Magenta=PnP", (10, legend_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
            print(f"  ✓ Saved PnP viz: {save_path}")
        
        return vis_img


def get_camera_matrix_from_fov(image_width, image_height, fov_radians):
    """
    Compute camera intrinsic matrix from FOV
    
    Args:
        image_width: Image width in pixels
        image_height: Image height in pixels
        fov_radians: Horizontal field of view in radians
        
    Returns:
        K: 3x3 camera matrix
    """
    # Focal length from horizontal FOV
    fx = (image_width / 2) / np.tan(fov_radians / 2)
    
    # Assume square pixels (typical for Gaussian splatting)
    fy = fx
    
    # Principal point at image center
    cx = image_width / 2
    cy = image_height / 2
    
    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float32)
    
    return K


if __name__ == "__main__":
    print("Window PnP estimator module loaded")
    print("Usage: from window_pnp import WindowPnPEstimator, get_camera_matrix_from_fov")