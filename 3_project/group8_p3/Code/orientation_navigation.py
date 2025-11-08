import numpy as np
import cv2
from navigation import goToWaypoint


def calculate_window_orientation_error(window_detection, img_center):
    """
    Calculate orientation mismatch between camera and window
    
    Uses window corner geometry to detect if viewing at an angle.
    
    Parameters:
    - window_detection: dict with 'corners' key (4 corner points)
    - img_center: [cx, cy] image center
    
    Returns:
    - yaw_error: horizontal rotation needed (radians, + = turn right)
    - pitch_error: vertical rotation needed (radians, + = tilt up)
    - is_frontal: bool, True if window appears rectangular (facing head-on)
    """
    
    corners = window_detection['corners']  # [TL, TR, BR, BL]
    
    # Calculate edge lengths
    top_width = np.linalg.norm(corners[1] - corners[0])
    bottom_width = np.linalg.norm(corners[2] - corners[3])
    left_height = np.linalg.norm(corners[3] - corners[0])
    right_height = np.linalg.norm(corners[2] - corners[1])
    
    # Calculate skew ratios (1.0 = perfect rectangle)
    width_ratio = min(top_width, bottom_width) / max(top_width, bottom_width)
    height_ratio = min(left_height, right_height) / max(left_height, right_height)
    
    # Check if frontal (rectangular appearance)
    is_frontal = width_ratio > 0.95 and height_ratio > 0.95
    
    # Calculate yaw error from width trapezoid
    if top_width > bottom_width * 1.1:
        # Top wider = window tilted away at top = need to tilt down
        yaw_direction = -1
    elif bottom_width > top_width * 1.1:
        # Bottom wider = window tilted toward us at bottom = need to tilt up
        yaw_direction = 1
    else:
        yaw_direction = 0
    
    # Calculate yaw magnitude from width difference
    width_diff_ratio = abs(top_width - bottom_width) / max(top_width, bottom_width)
    yaw_error_magnitude = width_diff_ratio * 0.5  # Scale to radians (rough estimate)
    
    # Calculate pitch error from height trapezoid
    if left_height > right_height * 1.1:
        # Left taller = window rotated left = need to turn right
        pitch_direction = 1
    elif right_height > left_height * 1.1:
        # Right taller = window rotated right = need to turn left
        pitch_direction = -1
    else:
        pitch_direction = 0
    
    # Calculate pitch magnitude
    height_diff_ratio = abs(left_height - right_height) / max(left_height, right_height)
    pitch_error_magnitude = height_diff_ratio * 0.5
    
    # Also use window center offset for additional yaw hint
    window_center = window_detection['center']
    horizontal_offset = window_center[0] - img_center[0]
    
    # If window is off-center AND trapezoidal, that indicates rotation
    fov_horizontal = 1.3089969  # radians (75 degrees)
    pixels_per_radian = img_center[0] * 2 / fov_horizontal
    center_yaw_error = horizontal_offset / pixels_per_radian
    
    # Combine geometric and position-based estimates
    yaw_error = yaw_direction * yaw_error_magnitude + center_yaw_error * 0.5
    pitch_error = pitch_direction * pitch_error_magnitude
    
    return yaw_error, pitch_error, is_frontal


def visualize_orientation_analysis(image, window_detection, yaw_error, pitch_error, is_frontal):
    """
    Draw orientation analysis on image
    """
    vis = image.copy()
    corners = window_detection['corners']
    
    # Draw corners with labels
    labels = ['TL', 'TR', 'BR', 'BL']
    for i, (corner, label) in enumerate(zip(corners, labels)):
        pt = tuple(corner.astype(int))
        cv2.circle(vis, pt, 8, (255, 0, 0), -1)
        cv2.putText(vis, label, (pt[0]+10, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 0), 2)
    
    # Draw edges with lengths
    edges = [
        (corners[0], corners[1], 'top'),
        (corners[2], corners[3], 'bottom'),
        (corners[0], corners[3], 'left'),
        (corners[1], corners[2], 'right')
    ]
    
    for p1, p2, name in edges:
        length = np.linalg.norm(p2 - p1)
        midpoint = ((p1 + p2) / 2).astype(int)
        cv2.line(vis, tuple(p1.astype(int)), tuple(p2.astype(int)), (0, 255, 0), 2)
        cv2.putText(vis, f'{length:.0f}px', tuple(midpoint), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    # Status overlay
    y_offset = 30
    status = "FRONTAL" if is_frontal else "ANGLED"
    color = (0, 255, 0) if is_frontal else (0, 165, 255)
    
    cv2.putText(vis, f"Orientation: {status}", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    y_offset += 30
    cv2.putText(vis, f"Yaw error: {np.degrees(yaw_error):+.1f} deg", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    y_offset += 30
    cv2.putText(vis, f"Pitch error: {np.degrees(pitch_error):+.1f} deg", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw rotation arrows
    if abs(yaw_error) > np.radians(5):
        arrow_text = "TURN RIGHT" if yaw_error > 0 else " TURN LEFT"
        cv2.putText(vis, arrow_text, (10, y_offset + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    return vis


def compute_orientation_correction_target(currentPose, yaw_error, pitch_error, 
                                         max_rotation_per_step=np.radians(10)):
    """
    Compute target pose for orientation correction (rotation only, no translation)
    
    Parameters:
    - currentPose: dict with 'position' and 'rpy' 
    - yaw_error: rotation needed around Z axis (radians)
    - pitch_error: rotation needed around Y axis (radians)
    - max_rotation_per_step: limit rotation per iteration (radians)
    
    Returns:
    - target_position: same as current (no translation)
    - target_rpy: corrected orientation
    - rotation_magnitude: total rotation being applied (radians)
    """
    
    # Limit rotation per step for stability
    yaw_correction = np.clip(yaw_error, -max_rotation_per_step, max_rotation_per_step)
    pitch_correction = np.clip(pitch_error, -max_rotation_per_step, max_rotation_per_step)
    
    # Apply corrections to current orientation
    current_rpy = currentPose['rpy'].copy()
    target_rpy = current_rpy.copy()
    
    target_rpy[2] += yaw_correction    # Yaw (Z-axis)
    target_rpy[1] += pitch_correction  # Pitch (Y-axis)
    target_rpy[0] = 0.0                # Keep roll at 0 (stay level)
    
    # Keep same position (rotate in place)
    target_position = currentPose['position'].copy()
    
    rotation_magnitude = np.sqrt(yaw_correction**2 + pitch_correction**2)
    
    return target_position, target_rpy, rotation_magnitude


def navigate_with_orientation_correction(renderer, currentPose, segmentor, detector, 
                                        windowCount, max_iterations=30):
    """
    Navigate with proper orientation correction BEFORE position correction
    
    Algorithm:
    1. Detect window
    2. Calculate orientation error (yaw, pitch)
    3. If orientation error > threshold: ROTATE (no translation)
    4. If oriented correctly: MOVE FORWARD
    5. Repeat until close enough to pass through
    """
    
    print(f"\n{'='*60}")
    print(f"Window {windowCount + 1} - Orientation-Aware Navigation")
    print(f"{'='*60}")
    
    ORIENTATION_THRESHOLD = np.radians(30)  # 5 degrees
    CLOSE_AREA_THRESHOLD = 0.50  # 25% of image
    
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
        print(f"Position: {currentPose['position']}")
        print(f"Orientation (rpy): {np.degrees(currentPose['rpy'])} degrees")
        
        # Render current view
        color_image, depth_image, metric_depth = renderer.render(
            currentPose['position'], 
            currentPose['rpy']
        )
        
        # Segment
        segmented = segmentor.get_pred(color_image)
        segmented = cv2.normalize(segmented, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Detect
        detections = detector.process_segmentation(segmented)
        
        # Save images
        iter_prefix = f'./log/window_{windowCount}_iter_{iteration:02d}'
        cv2.imwrite(f'{iter_prefix}_rgb.png', cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f'{iter_prefix}_segmentation.png', segmented)
        
        if  not detections:
            print(" No windows detected")
            
            # If we've lost sight of the window, try to scan
            if iteration < max_iterations - 2:
                print("  Lost window - scanning for target...")
                # Move smaller distance and maintain orientation
                target_pos = currentPose['position'].copy()
                target_pos[0] += 0.05  # Small forward motion while scanning
                targetPose = {
                    'position': target_pos,
                    'rpy': currentPose['rpy'].copy()
                }
                currentPose = goToWaypoint(currentPose, targetPose, velocity=0.05)  # Slower for safety
                continue
            else:
                return False, currentPose
        
        # Get closest window
        closest = detector.get_closest_window(detections)
        area_pct = closest['area'] / detector.image_area * 100
        
        print(f"Window area: {area_pct:.1f}%")
        
        # Calculate orientation error
        yaw_error, pitch_error, is_frontal = calculate_window_orientation_error(
            closest, detector.img_center
        )
        
        orientation_error_magnitude = np.sqrt(yaw_error**2 + pitch_error**2)
        
        print(f"Orientation error: {np.degrees(orientation_error_magnitude):.1f} "
              f"(yaw: {np.degrees(yaw_error):+.1f}°, pitch: {np.degrees(pitch_error):+.1f}°)")
        print(f"Frontal view: {is_frontal}")
        
        # Visualize with orientation info
        vis_img = visualize_orientation_analysis(color_image, closest, yaw_error, pitch_error, is_frontal)
        cv2.imwrite(f'{iter_prefix}_detection.png', cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        
        # Decision: Rotate or Move?
        if orientation_error_magnitude > ORIENTATION_THRESHOLD:
            print("this shouldnet happen, if your seeing this we are accounting for yaw which we shouldent be doing")
            # Need to correct orientation first
            print(f"Correcting orientation (error: {np.degrees(orientation_error_magnitude):.1f}°)...")
            
            # Keep same position but update orientation
            target_rpy = currentPose['rpy'].copy()
            
            # Apply orientation corrections directly
            target_rpy[2] += np.clip(yaw_error, -np.radians(5), np.radians(5))    # Yaw correction
            target_rpy[1] += np.clip(pitch_error, -np.radians(5), np.radians(5))  # Pitch correction
            target_rpy[0] = 0.0  # Keep roll level
            
            print(f"  Current RPY: {np.degrees(currentPose['rpy'])}")
            print(f"  Target RPY: {np.degrees(target_rpy)}")
            
            # Execute pure rotation (keep same position)
            targetPose = {
                'position': currentPose['position'].copy(),
                'rpy': target_rpy
            }
            currentPose = goToWaypoint(currentPose, targetPose, velocity=0.05)
            
        elif area_pct > CLOSE_AREA_THRESHOLD and is_frontal:
            # Oriented correctly AND close enough
            print(f" Oriented correctly and close ({area_pct:.1f}%) - FLYING THROUGH!")
            
            # Use window's position to compute target
            target_pos = detector.compute_navigation_target(
                currentPose['position'],
                closest,
                forward_distance=2.0  # Move through the window
            )
            
            targetPose = {
                'position': target_pos,
                'rpy': np.radians([0.0, 0.0, 0.0])  # Keep level while traversing
            }
            currentPose = goToWaypoint(currentPose, targetPose, velocity=0.6)
            
            return True, currentPose
            
        else:
            # Oriented correctly but not close yet - move toward window center
            print(f"Oriented correctly, moving toward window...")
            
            # Compute target based on window position
            target_pos = detector.compute_navigation_target(
                currentPose['position'],
                closest,
                forward_distance=0.3  # Smaller steps during approach
            )
            
            targetPose = {
                'position': target_pos,
                'rpy': np.radians([0.0, 0.0, 0.0])
            }
            currentPose = goToWaypoint(currentPose, targetPose, velocity=0.2)
    
    print(f" Max iterations reached")
    return False, currentPose