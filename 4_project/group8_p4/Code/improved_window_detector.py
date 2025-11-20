"""
Improved Window Detector with Bounding Box Selection
Fixes the detection to properly identify LOW flow regions (windows) and select the best bounding box
"""

import torch
import numpy as np
import cv2
from scanning_fix import compute_accumulated_flow_ts2p


class ImprovedWindowDetector:
    """
    Enhanced window detector that:
    1. Correctly identifies LOW flow regions as windows (closer objects)
    2. Finds connected components
    3. Selects the largest valid bounding box
    """
    
    def __init__(self, base_detector):
        """
        Args:
            base_detector: Your existing WindowDetector instance
        """
        self.base_detector = base_detector
        self.device = base_detector.device
        self.gap_detector = base_detector.gap_detector
        self.flow_extractor = base_detector.flow_extractor
        self.detection_resolution = base_detector.detection_resolution
    
    def detect_window_improved(self, frames):
        """
        Improved detection using SIMPLE optical flow (like simple_detector)
        
        Args:
            frames: List of (H, W, 3) RGB numpy arrays
        Returns:
            mask: (H, W) binary mask at ORIGINAL resolution
            center: (x, y) at ORIGINAL resolution
            confidence: [0, 1]
            debug_info: dict with intermediate results
        """
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
        
        print(f"  Computing TS²P accumulated flow (all consecutive pairs)...")
        
        # Use the optimized accumulated flow function
        Xi_np = compute_accumulated_flow_ts2p(downsampled_frames, 
                                              self.flow_extractor, 
                                              self.device)
        
        print(f"    Flow range: [{Xi_np.min():.2f}, {Xi_np.max():.2f}]")
        print(f"    Flow mean: {Xi_np.mean():.2f}")
        print(f"    Flow median: {np.median(Xi_np):.2f}")
        
        # Adaptive threshold strategy for MIN flow:
        # MIN flow creates much cleaner separation, so we need LOWER percentiles
        threshold_low = np.percentile(Xi_np, 10)   # Slightly more inclusive
        threshold_high = np.percentile(Xi_np, 20)  # Capture both windows fully
        
        print(f"    Low threshold (10%): {threshold_low:.2f}")
        print(f"    High threshold (20%): {threshold_high:.2f}")
        
        # Use the higher threshold to capture both windows
        threshold_val = threshold_high
        
        # Safety bounds - raise upper bound to capture flow ~6-7
        if threshold_val > 8:
            print(f"    WARNING: Threshold too high ({threshold_val:.2f}), capping at 8")
            threshold_val = 8
        elif threshold_val < 4:
            print(f"    WARNING: Threshold too low ({threshold_val:.2f}), raising to 4")
            threshold_val = 4
        
        print(f"    Using threshold: {threshold_val:.2f}")
        
        # Binary mask: LOW flow = windows
        chi_np = (Xi_np < threshold_val).astype(np.float32)
        
        print(f"    Pixels below threshold: {np.sum(chi_np > 0):.0f}")
        
        # Find connected components and select best bounding box
        mask_refined = self.select_best_bounding_box(chi_np)
        
        # Upsample back to original resolution
        mask = cv2.resize(mask_refined, (original_W, original_H), 
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
        
        # Debug info
        debug_info = {
            'Xi': Xi_np,
            'threshold': threshold_val,
            'chi_before': chi_np,
            'chi_after': mask_refined,
            'frames': downsampled_frames
        }
        
        print(f"  Threshold: {threshold_val:.2f}")
        print(f"  Pixels before bbox selection: {np.sum(chi_np > 0.5):.0f}")
        print(f"  Pixels after bbox selection: {np.sum(mask_refined > 0.5):.0f}")
        print(f"  Confidence: {confidence:.3f}")
        
        return mask, center, confidence, debug_info
    
    def select_best_bounding_box(self, mask):
        """
        Select the best bounding box from LOW flow regions (blue regions in flow map)
        
        Strategy:
        1. REMOVE FLOOR FIRST (bottom 40% of image)
        2. Light morphology to clean noise
        3. Find connected components
        4. Filter (edges, size, aspect ratio)
        5. Select LARGEST valid window
        
        Args:
            mask: (H, W) binary mask
        Returns:
            result_mask: (H, W) refined binary mask
        """
        H, W = mask.shape
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Step 0: CRITICAL - Remove floor region BEFORE morphology
        # Floor is typically in bottom 40% of image
        floor_cutoff = int(H * 0.6)  # Keep only top 60%
        mask_no_floor = mask_uint8.copy()
        mask_no_floor[floor_cutoff:, :] = 0  # Zero out bottom 40%
        
        print(f"    Removed floor (bottom 40%, {H - floor_cutoff}px)")
        print(f"    Pixels after floor removal: {np.sum(mask_no_floor > 0)}")
        
        # Step 1: Light morphology to clean noise (like simple detector)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_clean = cv2.morphologyEx(mask_no_floor, cv2.MORPH_OPEN, kernel_open)
        
        mask_uint8 = mask_clean
        
        # Step 2: Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_uint8, connectivity=8)
        
        print(f"    Found {num_labels - 1} connected components")
        
        if num_labels <= 1:
            return np.zeros((H, W), dtype=np.float32)
        
        # Step 3: Filter components - relaxed for MIN flow (very clean signal)
        valid_regions = []
        for label_id in range(1, num_labels):
            x, y, w, h, area = stats[label_id]
            cx, cy = centroids[label_id]
            
            # 1. Size bounds - VERY RELAXED
            min_area = int(0.001 * H * W)  # At least 0.1% (was 0.3%)
            max_area = int(0.6 * H * W)    # At most 60% (was 50%)
            if area < min_area:
                print(f"      Component {label_id}: REJECTED (too small: {area})")
                continue
            if area > max_area:
                print(f"      Component {label_id}: REJECTED (too large: {area})")
                continue
            
            # 2. Edge rejection - RELAXED (10px margin)
            margin = 10  # Was 20px
            if x < margin or y < margin or x+w > W-margin or y+h > H-margin:
                print(f"      Component {label_id}: REJECTED (at edge)")
                continue
            
            # 3. Aspect ratio - VERY RELAXED
            aspect_ratio = w / (h + 1e-6)
            if aspect_ratio < 0.1 or aspect_ratio > 10.0:  # Was 0.2 to 5.0
                print(f"      Component {label_id}: REJECTED (bad aspect: {aspect_ratio:.2f})")
                continue
            
            # 4. Compactness check - RELAXED
            bbox_area = w * h
            if bbox_area > 0:
                compactness = area / bbox_area
                if compactness < 0.2:  # Was 0.3
                    print(f"      Component {label_id}: REJECTED (low compactness: {compactness:.2f})")
                    continue
            
            # Score: larger area + higher position = better
            vertical_score = 1.0 - (cy / H)  # Higher in image = better
            img_center_x = W / 2
            horizontal_centrality = 1.0 - abs(cx - img_center_x) / (W / 2)
            
            # Prioritize: size (60%), horizontal centrality (25%), height (15%)
            score = area * (0.6 + 0.25 * horizontal_centrality + 0.15 * vertical_score)
            
            print(f"      Component {label_id}: VALID (area={area}, cy={cy:.0f}, score={score:.0f})")
            
            valid_regions.append({
                'label_id': label_id,
                'score': score,
                'bbox': (x, y, w, h),
                'area': area
            })
        
        # Step 4: Select highest scoring region
        if not valid_regions:
            print("    No valid regions found")
            return np.zeros((H, W), dtype=np.float32)
        
        valid_regions.sort(key=lambda x: x['score'], reverse=True)
        best_region = valid_regions[0]
        best_label = best_region['label_id']
        
        print(f"    Selected component {best_label} (area={best_region['area']}, bbox={best_region['bbox']})")
        
        # Create mask from selected component
        result_mask = (labels == best_label).astype(np.uint8) * 255
        
        # Step 5: Light closing to smooth
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        result_mask = cv2.morphologyEx(result_mask, cv2.MORPH_CLOSE, kernel_close)
        
        return result_mask / 255.0
    
    def visualize_detection_process(self, debug_info, save_path='./log/detection_process.png'):
        """
        Create comprehensive visualization of detection process
        
        Args:
            debug_info: dict from detect_window_improved
            save_path: where to save visualization
        """
        import matplotlib.pyplot as plt
        
        Xi = debug_info['Xi']
        threshold = debug_info['threshold']
        chi_before = debug_info['chi_before']
        chi_after = debug_info['chi_after']
        frames = debug_info['frames']
        
        # Find bounding boxes
        mask_uint8 = (chi_before * 255).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_uint8, connectivity=8)
        
        # Create figure
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        
        # Row 1: Input frames
        axes[0, 0].imshow(frames[0])
        axes[0, 0].set_title('Frame 0 (Reference)', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(frames[2])
        axes[0, 1].set_title('Frame 2 (Middle)', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(frames[4])
        axes[0, 2].set_title('Frame 4 (Last)', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Row 2: Flow analysis
        im1 = axes[1, 0].imshow(Xi, cmap='jet')
        axes[1, 0].set_title(f'Flow Magnitude (Ξ)\nBLUE = LOW flow (windows)', 
                            fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
        
        # Histogram
        axes[1, 1].hist(Xi.flatten(), bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(threshold, color='r', linestyle='--', linewidth=2, 
                        label=f'Threshold: {threshold:.1f}')
        axes[1, 1].set_xlabel('Flow Magnitude', fontsize=10)
        axes[1, 1].set_ylabel('Pixel Count', fontsize=10)
        axes[1, 1].set_title('Flow Distribution', fontsize=12, fontweight='bold')
        axes[1, 1].legend(fontsize=9)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Bounding boxes visualization
        frame_with_boxes = frames[0].copy()
        for label_id in range(1, num_labels):
            x, y, w, h, area = stats[label_id]
            # Color code: green for large, yellow for medium, red for small
            if area > 1000:
                color = (0, 255, 0)
            elif area > 500:
                color = (255, 255, 0)
            else:
                color = (255, 0, 0)
            cv2.rectangle(frame_with_boxes, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame_with_boxes, f'{area}', (x+2, y+12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        axes[1, 2].imshow(frame_with_boxes)
        axes[1, 2].set_title(f'All Bounding Boxes\n{num_labels-1} regions', 
                            fontsize=12, fontweight='bold')
        axes[1, 2].axis('off')
        
        # Row 3: Detection stages
        axes[2, 0].imshow(chi_before, cmap='gray', vmin=0, vmax=1)
        axes[2, 0].set_title(f'After Threshold\n{np.sum(chi_before > 0.5):.0f} pixels', 
                            fontsize=12, fontweight='bold')
        axes[2, 0].axis('off')
        
        axes[2, 1].imshow(chi_after, cmap='gray', vmin=0, vmax=1)
        axes[2, 1].set_title(f'After Bbox Selection\n{np.sum(chi_after > 0.5):.0f} pixels', 
                            fontsize=12, fontweight='bold')
        axes[2, 1].axis('off')
        
        # Final overlay
        overlay = frames[0].copy()
        mask_overlay = np.zeros_like(overlay)
        mask_overlay[chi_after > 0.5] = [0, 255, 0]
        overlay = cv2.addWeighted(overlay, 0.7, mask_overlay, 0.3, 0)
        
        # Draw final bounding box and arrow
        if np.sum(chi_after > 0.5) > 0:
            y_coords, x_coords = np.where(chi_after > 0.5)
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()
            
            # Bounding box (blue)
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)
            
            # Window center (red circle)
            cx = (x_min + x_max) // 2
            cy = (y_min + y_max) // 2
            cv2.circle(overlay, (cx, cy), 10, (255, 0, 0), -1)
            
            # Image center (cyan cross)
            H_viz, W_viz = overlay.shape[:2]
            img_cx = W_viz // 2
            img_cy = H_viz // 2
            cv2.drawMarker(overlay, (img_cx, img_cy), (255, 255, 0), 
                          markerType=cv2.MARKER_CROSS, 
                          markerSize=30, thickness=3)
            
            # Arrow from image center to window center (yellow)
            cv2.arrowedLine(overlay, (img_cx, img_cy), (cx, cy), 
                           (0, 255, 255), 3, tipLength=0.05)
            
            # Distance text
            distance_px = np.sqrt((cx - img_cx)**2 + (cy - img_cy)**2)
            cv2.putText(overlay, f'Offset: {distance_px:.0f}px', (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        axes[2, 2].imshow(overlay)
        axes[2, 2].set_title('Final Detection\n(Blue box = selected window)', 
                            fontsize=12, fontweight='bold')
        axes[2, 2].axis('off')
        
        plt.suptitle('TS²P Window Detection: LOW Flow = Window (Closer Object)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved visualization to {save_path}")
        plt.close()