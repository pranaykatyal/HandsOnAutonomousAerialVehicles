"""
Improved Window Detector with Bounding Box Selection
Fixes the detection to properly identify LOW flow regions (windows) and select the best bounding box
"""

import torch
import numpy as np
import cv2


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
        Improved detection with bounding box selection
        
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
        
        # Convert to torch tensor
        frames_np = np.array(downsampled_frames)
        frames_tensor = torch.from_numpy(frames_np).float() / 255.0
        frames_tensor = frames_tensor.permute(0, 3, 1, 2)
        frames_tensor = frames_tensor.unsqueeze(0).to(self.device)
        
        print(f"  Running TS²P detection...")
        
        with torch.no_grad():
            # Compute flows
            F_stack = self.gap_detector.compute_temporal_flow_stack(frames_tensor)
            Xi = self.gap_detector.compute_flow_magnitude(F_stack)
            
            # Get threshold value
            B, _, H, W = Xi.shape
            threshold = torch.quantile(Xi.reshape(B, -1), 
                                    q=self.gap_detector.threshold_percentile/100.0, 
                                    dim=1, keepdim=True)
            threshold_val = threshold.item()
            
            # CORRECT LOGIC: LOW flow = window (closer object)
            chi = (Xi < threshold).float()
        
        # Convert to numpy
        Xi_np = Xi[0, 0].cpu().numpy()
        chi_np = chi[0, 0].cpu().numpy()
        
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
        1. Light morphology to clean noise
        2. Find connected components
        3. Filter invalid components (too small, at edges, wrong aspect ratio)
        4. Select LARGEST valid component
        5. Apply light closing to smooth result
        
        Args:
            mask: (H, W) binary mask
        Returns:
            result_mask: (H, W) refined binary mask
        """
        H, W = mask.shape
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Step 1: Light morphological opening to remove small noise
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel_small)
        
        # Step 2: Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_uint8, connectivity=8)
        
        print(f"    Found {num_labels - 1} connected components")
        
        if num_labels <= 1:  # No foreground regions
            return np.zeros((H, W), dtype=np.float32)
        
        # Step 3: Evaluate each component (skip background label 0)
        valid_regions = []
        for label_id in range(1, num_labels):
            x, y, w, h, area = stats[label_id]
            
            # Validity checks
            # 1. Minimum size (at least 500 pixels for 512x512 image)
            min_area = int(0.002 * H * W)  # 0.2% of image
            if area < min_area:
                print(f"      Component {label_id}: REJECTED (too small: {area} < {min_area})")
                continue
            
            # 2. Not too close to edges (20 pixel margin)
            margin = 20
            if x < margin or y < margin or x+w > W-margin or y+h > H-margin:
                print(f"      Component {label_id}: REJECTED (at edge)")
                continue
            
            # 3. Reasonable aspect ratio (not too elongated)
            aspect_ratio = w / (h + 1e-6)
            if aspect_ratio < 0.15 or aspect_ratio > 6.0:
                print(f"      Component {label_id}: REJECTED (bad aspect: {aspect_ratio:.2f})")
                continue
            
            # 4. Reasonable size (not too large - probably artifacts)
            if area > 0.8 * H * W:
                print(f"      Component {label_id}: REJECTED (too large)")
                continue
            
            # Compute score: larger area + more central = better
            cx, cy = centroids[label_id]
            img_center_x, img_center_y = W / 2, H / 2
            dist_from_center = np.sqrt((cx - img_center_x)**2 + (cy - img_center_y)**2)
            max_dist = np.sqrt((W/2)**2 + (H/2)**2)
            centrality_score = 1.0 - (dist_from_center / max_dist)
            
            # Combined score: area is primary (70%), centrality is secondary (30%)
            score = area * (0.7 + 0.3 * centrality_score)
            
            print(f"      Component {label_id}: VALID (area={area}, score={score:.0f})")
            
            valid_regions.append({
                'label_id': label_id,
                'score': score,
                'bbox': (x, y, w, h),
                'area': area
            })
        
        # Step 4: Select best region
        if not valid_regions:
            print("    No valid regions found, using largest component")
            # Fallback: use largest component ignoring validity checks
            areas = stats[1:, cv2.CC_STAT_AREA]  # Skip background
            if len(areas) > 0:
                best_label = np.argmax(areas) + 1
                result_mask = (labels == best_label).astype(np.uint8) * 255
            else:
                result_mask = np.zeros((H, W), dtype=np.uint8)
        else:
            # Use highest scoring region
            valid_regions.sort(key=lambda x: x['score'], reverse=True)
            best_region = valid_regions[0]
            best_label = best_region['label_id']
            
            print(f"    Selected component {best_label} (area={best_region['area']}, bbox={best_region['bbox']})")
            
            # Create mask from selected component
            result_mask = (labels == best_label).astype(np.uint8) * 255
        
        # Step 5: Apply light closing to fill small holes within selected region
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
        
        # Draw final bounding box
        if np.sum(chi_after > 0.5) > 0:
            y_coords, x_coords = np.where(chi_after > 0.5)
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)
            # Draw center
            cx = (x_min + x_max) // 2
            cy = (y_min + y_max) // 2
            cv2.circle(overlay, (cx, cy), 10, (255, 0, 0), -1)
        
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


# Example integration with your existing code:
"""
# In your main.py, replace detector.detect_window() with:

from improved_window_detector import ImprovedWindowDetector

# Initialize
detector = WindowDetector(device='cuda')
improved_detector = ImprovedWindowDetector(detector)

# Use improved detection
mask, center, confidence, debug_info = improved_detector.detect_window_improved(scan_frames)

# Visualize the process
improved_detector.visualize_detection_process(debug_info)
"""