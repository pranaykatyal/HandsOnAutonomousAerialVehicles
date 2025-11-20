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
        
        # Compute confidence using downsampled refined mask + flow map
        conf, conf_comp = self.compute_confidence(mask_refined, Xi_np)
        
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
        else:
            center = None
        
        # Keep backward-compatible confidence behavior: ensure numeric in [0,1]
        confidence = float(np.clip(conf, 0.0, 1.0))
        
        # Debug info
        debug_info = {
            'Xi': Xi_np,
            'threshold': threshold_val,
            'chi_before': chi_np,
            'chi_after': mask_refined,
            'frames': downsampled_frames,
            'confidence_components': conf_comp
        }
        
        print(f"  Threshold: {threshold_val:.2f}")
        print(f"  Pixels before bbox selection: {np.sum(chi_np > 0.5):.0f}")
        print(f"  Pixels after bbox selection: {np.sum(mask_refined > 0.5):.0f}")
        print(f"  Confidence: {confidence:.3f} (sol={conf_comp['solidity']:.3f}, ar={conf_comp['aspect']:.3f}, contrast={conf_comp['contrast']:.3f})")
        
        return mask, center, confidence, debug_info
    
    def compute_confidence(self, mask_down, Xi_np):
        """
        Compute a distance-invariant confidence for a detected mask.

        Inputs:
          - mask_down: (H, W) binary mask at detection_resolution (values 0..1)
          - Xi_np: (H, W) flow magnitude map used to compute mask_down

        Returns:
          - confidence: float in [0,1]
          - components: dict with sub-scores for debugging
        Principles:
          - Do NOT penalize large masks just for being large.
          - Use region solidity (area / convex-hull-area).
          - Use aspect-ratio score (windows usually have reasonable AR).
          - Use local contrast: mean_flow_outside - mean_flow_inside (normalized).
        """
        # Convert to uint8 mask
        H, W = mask_down.shape
        eps = 1e-9
        mask_u8 = (mask_down > 0.5).astype(np.uint8)  # 0/1
        mask_u8_255 = (mask_u8 * 255).astype(np.uint8)
        
        # Default (no detection)
        if mask_u8.sum() == 0:
            return 0.0, {'solidity': 0.0, 'aspect': 0.0, 'contrast': 0.0}
        
        # Find contours and pick the largest contour by area
        contours, _ = cv2.findContours(mask_u8_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0, {'solidity': 0.0, 'aspect': 0.0, 'contrast': 0.0}
        
        # choose the biggest contour
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area <= 0:
            return 0.0, {'solidity': 0.0, 'aspect': 0.0, 'contrast': 0.0}
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = max(1, w * h)
        
        # Solidity: area / convex hull area
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            solidity = 0.0
        else:
            solidity = float(area / (hull_area + eps))
            solidity = float(np.clip(solidity, 0.0, 1.0))
        
        # Aspect ratio score: prefer AR in [0.4, 2.5] but smoothly decay otherwise
        ar = (w / (h + eps))
        # map ar to score in (0,1]
        # when ar==1 ->1.0; as |log(ar)| grows, score decreases
        ar_log = abs(np.log(ar + eps))
        aspect_score = float(np.clip(1.0 - (ar_log / 2.0), 0.0, 1.0))
        
        # Local contrast: compare mean flow inside mask vs in a surrounding ring
        # Build ring mask: expanded bbox minus the mask itself
        pad = int(max(w, h) * 0.5) + 5
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(W, x + w + pad)
        y1 = min(H, y + h + pad)
        
        ring_mask = np.zeros((H, W), dtype=np.uint8)
        ring_mask[y0:y1, x0:x1] = 1
        # zero inner bbox region so ring excludes the bbox interior
        ring_mask[y:y+h, x:x+w] = 0
        # exclude existing detected mask pixels from ring (so ring is truly outside)
        ring_mask = ring_mask & (1 - mask_u8)
        
        inside_vals = Xi_np[mask_u8.astype(bool)]
        outside_vals = Xi_np[ring_mask.astype(bool)]
        
        mean_inside = float(np.mean(inside_vals)) if inside_vals.size > 0 else float(np.mean(Xi_np))
        mean_outside = float(np.mean(outside_vals)) if outside_vals.size > 0 else float(np.mean(Xi_np))
        
        # We expect outside > inside (higher flow outside the low-flow window).
        # contrast_raw in (-inf, +inf). Convert to [0,1] with sigmoid-like mapping.
        contrast_raw = (mean_outside - mean_inside) / (abs(mean_outside) + eps)
        # clip negative -> 0, since negative means inside isn't lower than outside
        contrast = float(np.clip(contrast_raw, 0.0, 1.0))
        
        # Combine components with weights
        # solidity: 35%, aspect: 25%, contrast: 40%
        conf = 0.35 * solidity + 0.25 * aspect_score + 0.40 * contrast
        
        # Small-area safeguard: if detected region extremely tiny relative to frame, downweight
        min_area = max(1, int(0.0005 * H * W))
        if area < min_area:
            conf *= 0.0  # too small, no confidence
        # clamp and return
        conf = float(np.clip(conf, 0.0, 1.0))
        
        components = {
            'solidity': float(solidity),
            'aspect': float(aspect_score),
            'contrast': float(contrast),
            'area_px': float(area),
            'bbox': (int(x), int(y), int(w), int(h)),
            'mean_inside_flow': float(mean_inside),
            'mean_outside_flow': float(mean_outside)
        }
        return conf, components
    
    def select_best_bounding_box(self, mask):
        """
        Select the best bounding box from LOW flow regions (blue regions in flow map)
        
        Behavior:
        - Run strict filtering first (edge-touching components rejected).
        - If no valid regions found, re-run filtering with edge constraint relaxed.
        
        Args:
            mask: (H, W) binary mask
        Returns:
            result_mask: (H, W) refined binary mask (values 0.0/1.0)
        """
        H, W = mask.shape
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Keep full mask (floor removal disabled)
        mask_no_floor = mask_uint8.copy()
        print(f"    Floor removal DISABLED - keeping full frame for bbox selection")
        print(f"    Pixels before morphology: {np.sum(mask_no_floor > 0)}")
        
        # Light morphology to clean noise
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_clean = cv2.morphologyEx(mask_no_floor, cv2.MORPH_OPEN, kernel_open)
        
        # Inner helper: filter components with optional edge allowance
        def _filter_and_score(mask_u8, allow_edge=False):
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask_u8, connectivity=8)
            
            valid_regions = []
            for label_id in range(1, num_labels):
                x, y, w, h, area = stats[label_id]
                cx, cy = centroids[label_id]
                
                # Basic size bounds (keep relatively permissive)
                min_area = int(0.001 * H * W)  # 0.1% of image
                max_area = int(0.95 * H * W)   # allow up to 95% (big windows when close)
                if area < min_area:
                    # too small
                    # print(f"      Component {label_id}: REJECTED (too small: {area})")
                    continue
                if area > max_area:
                    # too large (unlikely but safe)
                    # print(f"      Component {label_id}: REJECTED (too large: {area})")
                    continue
                
                # Edge rejection (strict mode): reject if touching image edges
                margin = 10
                touches_edge = (x < margin or y < margin or (x + w) > (W - margin) or (y + h) > (H - margin))
                if (not allow_edge) and touches_edge:
                    # rejected in strict mode
                    # print(f"      Component {label_id}: REJECTED (touches edge)")
                    continue
                
                # Aspect ratio filter (permissive)
                aspect_ratio = w / (h + 1e-6)
                if aspect_ratio < 0.05 or aspect_ratio > 20.0:
                    # print(f"      Component {label_id}: REJECTED (bad aspect: {aspect_ratio:.2f})")
                    continue
                
                # Compactness (allow concave / irregular shapes)
                bbox_area = w * h
                if bbox_area > 0:
                    compactness = area / bbox_area
                    if compactness < 0.05:
                        # print(f"      Component {label_id}: REJECTED (low compactness: {compactness:.3f})")
                        continue
                
                # Score: prefer larger area, slightly prefer central & higher components
                vertical_score = 1.0 - (cy / H)
                img_center_x = W / 2
                horizontal_centrality = 1.0 - abs(cx - img_center_x) / (W / 2)
                score = area * (0.6 + 0.25 * horizontal_centrality + 0.15 * vertical_score)
                
                valid_regions.append({
                    'label_id': label_id,
                    'score': score,
                    'bbox': (x, y, w, h),
                    'area': area
                })
            
            if not valid_regions:
                return None, None, None
            
            # pick best region
            valid_regions.sort(key=lambda r: r['score'], reverse=True)
            best = valid_regions[0]
            best_mask = (labels == best['label_id']).astype(np.uint8) * 255
            return best_mask, labels, valid_regions
        
        # First try: strict mode (do not allow edge-touching components)
        best_mask_strict, labels_strict, valid_strict = _filter_and_score(mask_clean, allow_edge=False)
        if best_mask_strict is not None:
            print(f"    select_best_bounding_box: strict mode succeeded ({len(valid_strict)} valid regions)")
            # smooth and return
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            result_mask = cv2.morphologyEx(best_mask_strict, cv2.MORPH_CLOSE, kernel_close)
            return (result_mask / 255.0).astype(np.float32)
        
        # Fallback: relax edge constraint (allow edge-touching blobs)
        print("    select_best_bounding_box: no valid regions in strict mode → relaxing edge constraint (fallback).")
        best_mask_relaxed, labels_relaxed, valid_relaxed = _filter_and_score(mask_clean, allow_edge=True)
        if best_mask_relaxed is not None:
            print(f"    select_best_bounding_box: fallback succeeded ({len(valid_relaxed)} valid regions)")
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            result_mask = cv2.morphologyEx(best_mask_relaxed, cv2.MORPH_CLOSE, kernel_close)
            return (result_mask / 255.0).astype(np.float32)
        
        # Still nothing
        print("    select_best_bounding_box: fallback also found no valid regions → returning empty mask")
        return np.zeros((H, W), dtype=np.float32)
    
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
