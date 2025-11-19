"""
Simple Optical Flow Window Detector
Directly uses the flow magnitude to find windows (blue/low-flow regions)
Much simpler than TS²P!
"""

import torch
import numpy as np
import cv2


class SimpleFlowDetector:
    """
    Ultra-simple window detector:
    1. Compute optical flow between first and last frame
    2. Find LOW flow regions (blue areas in your debug image)
    3. Select largest valid bounding box
    """
    
    def __init__(self, flow_extractor, device='cuda'):
        """
        Args:
            flow_extractor: Your OpticalFlowExtractor instance
            device: torch device
        """
        self.flow_extractor = flow_extractor
        self.device = device
        self.detection_resolution = 512
    
    def detect_window_simple(self, frames):
        """
        Simple flow-based detection
        
        Args:
            frames: List of (H, W, 3) RGB numpy arrays
        Returns:
            mask: (H, W) binary mask at ORIGINAL resolution
            center: (x, y) at ORIGINAL resolution
            confidence: [0, 1]
            debug_info: dict with intermediate results
        """
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
        
        # Convert to torch tensors
        frame_0 = torch.from_numpy(downsampled_frames[0]).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        frame_last = torch.from_numpy(downsampled_frames[-1]).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        
        frame_0 = frame_0.to(self.device)
        frame_last = frame_last.to(self.device)
        
        print(f"  Computing optical flow...")
        
        # Compute optical flow between first and last frame
        with torch.no_grad():
            flow = self.flow_extractor.compute_flow(frame_0, frame_last)  # (1, 2, H, W)
            
            # Flow magnitude
            u = flow[0, 0]  # Horizontal flow
            v = flow[0, 1]  # Vertical flow
            flow_magnitude = torch.sqrt(u**2 + v**2)  # (H, W)
        
        Xi_np = flow_magnitude.cpu().numpy()
        
        print(f"    Flow range: [{Xi_np.min():.2f}, {Xi_np.max():.2f}]")
        print(f"    Flow mean: {Xi_np.mean():.2f}")
        print(f"    Flow median: {np.median(Xi_np):.2f}")
        
        # Find LOW flow regions (windows)
        # Use a threshold that selects ONLY the lowest flow values
        # Since most of the scene has high flow (~100-120), we need to be very selective
        threshold_percentile = 5  # Select bottom 5% (ONLY the clearest windows)
        threshold = np.percentile(Xi_np, threshold_percentile)
        
        # Additional safeguard: if threshold is too high, cap it
        # Windows should have flow < 60, background is > 80
        if threshold > 60:
            print(f"    WARNING: Threshold too high ({threshold:.2f}), capping at 60")
            threshold = 60
        
        print(f"    Threshold ({threshold_percentile}%): {threshold:.2f}")
        
        # Binary mask: 1 where flow < threshold (LOW flow = windows)
        binary_mask = (Xi_np < threshold).astype(np.uint8) * 255
        
        print(f"    Pixels below threshold: {np.sum(binary_mask > 0):.0f}")
        
        # Select best bounding box
        mask_refined = self.select_best_bbox_simple(binary_mask)
        
        # Upsample back to original resolution
        mask = cv2.resize(mask_refined, (original_W, original_H), 
                         interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0.5).astype(np.float32)
        
        # Compute center
        if mask.sum() > 100:
            y_coords, x_coords = np.where(mask > 0.5)
            center_x = int(x_coords.mean())
            center_y = int(y_coords.mean())
            center = (center_x, center_y)
            
            mask_area = mask.sum()
            confidence = min(1.0, mask_area / (original_H * original_W * 0.3))
        else:
            center = None
            confidence = 0.0
        
        # Debug info
        debug_info = {
            'Xi': Xi_np,
            'threshold': threshold,
            'binary_mask': binary_mask,
            'mask_refined': mask_refined,
            'frames': downsampled_frames
        }
        
        print(f"  Final mask pixels: {np.sum(mask_refined > 0.5):.0f}")
        print(f"  Confidence: {confidence:.3f}")
        
        return mask, center, confidence, debug_info
    
    def select_best_bbox_simple(self, mask):
        """
        Find and select the best bounding box from LOW flow regions
        
        Args:
            mask: (H, W) uint8 binary mask (0 or 255)
        Returns:
            result_mask: (H, W) float32 mask (0.0 or 1.0)
        """
        H, W = mask.shape
        
        # Light morphology to clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_clean, connectivity=8)
        
        print(f"    Found {num_labels - 1} connected components")
        
        if num_labels <= 1:
            return np.zeros((H, W), dtype=np.float32)
        
        # Find valid components
        valid_components = []
        for label_id in range(1, num_labels):
            x, y, w, h, area = stats[label_id]
            
            # Filters - be MORE selective
            min_area = int(0.005 * H * W)  # At least 0.5% of image (was 0.1%)
            max_area = int(0.5 * H * W)     # At most 50% of image (new!)
            margin = 20  # Stay away from edges
            
            # Size check
            if area < min_area:
                print(f"      Component {label_id}: REJECTED (too small: {area})")
                continue
            
            if area > max_area:
                print(f"      Component {label_id}: REJECTED (too large: {area})")
                continue
            
            # Edge check - be stricter
            if x < margin or y < margin or x+w > W-margin or y+h > H-margin:
                print(f"      Component {label_id}: REJECTED (at edge)")
                continue
            
            # Position check: reject things in bottom half (likely floor)
            center_y = y + h/2
            if center_y > H * 0.7:  # Below 70% of image height
                print(f"      Component {label_id}: REJECTED (too low - likely floor)")
                continue
            
            # Aspect ratio check - windows are roughly rectangular
            aspect = w / (h + 1e-6)
            if aspect < 0.3 or aspect > 3.5:  # More reasonable range
                print(f"      Component {label_id}: REJECTED (bad aspect: {aspect:.2f})")
                continue
            
            # Score: bigger is better, more central is better, HIGHER is much better
            cx, cy = centroids[label_id]
            dist_from_center = np.sqrt((cx - W/2)**2 + (cy - H/2)**2)
            max_dist = np.sqrt((W/2)**2 + (H/2)**2)
            centrality = 1.0 - (dist_from_center / max_dist)
            
            # Vertical position score - STRONGLY prefer upper regions (windows not floor!)
            # cy = 0 (top) = 1.0, cy = H (bottom) = 0.0
            vertical_score = 1.0 - (cy / H)
            
            # Combined score: size + centrality + STRONG preference for upper regions
            score = area * (0.4 + 0.2 * centrality + 0.4 * vertical_score)
            #                ^^^    ^^^                 ^^^ IMPORTANT: prefer top of image!
            
            print(f"      Component {label_id}: VALID (area={area}, score={score:.0f})")
            
            valid_components.append({
                'label_id': label_id,
                'score': score,
                'area': area,
                'bbox': (x, y, w, h)
            })
        
        # Select best
        if not valid_components:
            print("    No valid components, using largest")
            areas = stats[1:, cv2.CC_STAT_AREA]
            if len(areas) > 0:
                best_label = np.argmax(areas) + 1
            else:
                return np.zeros((H, W), dtype=np.float32)
        else:
            valid_components.sort(key=lambda x: x['score'], reverse=True)
            best_label = valid_components[0]['label_id']
            print(f"    Selected component {best_label} (area={valid_components[0]['area']})")
        
        # Create mask
        result = (labels == best_label).astype(np.uint8) * 255
        
        # Light closing to smooth
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel_close)
        
        return result.astype(np.float32) / 255.0
    
    def visualize_simple(self, debug_info, save_path='./log/flow_detection.png'):
        """
        Visualize the simple flow-based detection
        """
        import matplotlib.pyplot as plt
        
        Xi = debug_info['Xi']
        threshold = debug_info['threshold']
        binary_mask = debug_info['binary_mask']
        mask_refined = debug_info['mask_refined']
        frames = debug_info['frames']
        
        # Find components for visualization
        mask_uint8 = binary_mask.astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Row 1: Input and flow
        axes[0, 0].imshow(frames[0])
        axes[0, 0].set_title('First Frame', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(frames[-1])
        axes[0, 1].set_title('Last Frame', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        im = axes[0, 2].imshow(Xi, cmap='jet')
        axes[0, 2].set_title(f'Flow Magnitude\nBLUE = Window (LOW flow)', 
                            fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')
        plt.colorbar(im, ax=axes[0, 2], fraction=0.046)
        
        # Row 2: Detection stages
        axes[1, 0].hist(Xi.flatten(), bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(threshold, color='r', linestyle='--', linewidth=2,
                          label=f'Threshold: {threshold:.1f}')
        axes[1, 0].set_xlabel('Flow Magnitude', fontsize=11)
        axes[1, 0].set_ylabel('Pixel Count', fontsize=11)
        axes[1, 0].set_title('Flow Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Show all bounding boxes
        frame_boxes = frames[0].copy()
        for label_id in range(1, num_labels):
            x, y, w, h, area = stats[label_id]
            color = (0, 255, 0) if area > 500 else (255, 0, 0)
            cv2.rectangle(frame_boxes, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame_boxes, f'{area}', (x+2, y+12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        axes[1, 1].imshow(frame_boxes)
        axes[1, 1].set_title(f'All Bounding Boxes\n{num_labels-1} regions found', 
                            fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        # Final detection overlay
        overlay = frames[0].copy()
        mask_overlay = np.zeros_like(overlay)
        mask_overlay[mask_refined > 0.5] = [0, 255, 0]
        overlay = cv2.addWeighted(overlay, 0.7, mask_overlay, 0.3, 0)
        
        # Draw bounding box
        if np.sum(mask_refined > 0.5) > 0:
            y_coords, x_coords = np.where(mask_refined > 0.5)
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)
            # Center
            cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
            cv2.circle(overlay, (cx, cy), 10, (255, 0, 0), -1)
        
        axes[1, 2].imshow(overlay)
        axes[1, 2].set_title(f'Final Detection\n{np.sum(mask_refined > 0.5):.0f} pixels', 
                            fontsize=14, fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.suptitle('Simple Optical Flow Window Detection', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved visualization to {save_path}")
        plt.close()


# Integration with your existing code
"""
from simple_flow_detector import SimpleFlowDetector

# In main(), after initializing detector:
simple_detector = SimpleFlowDetector(detector.flow_extractor, device='cuda')

# Detect:
window_mask, window_center_2d, confidence, debug_info = simple_detector.detect_window_simple(scan_frames)

# Visualize:
simple_detector.visualize_simple(debug_info, './log/simple_detection.png')
"""