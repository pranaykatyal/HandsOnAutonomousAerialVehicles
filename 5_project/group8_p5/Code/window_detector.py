"""
Window Detection using Optical Flow (from Project 4)
Integrates RAFT optical flow for TS²P-style window detection
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'RAFT/core'))

import torch
import numpy as np
import cv2
from raft import RAFT
from argparse import Namespace


class OpticalFlowExtractor:
    """RAFT-based optical flow computation"""
    
    def __init__(self, model_path='./RAFT/models/raft-things.pth', device='cuda'):
        self.device = device
        
        # RAFT args
        args = Namespace(
            model=model_path,
            small=False,
            mixed_precision=True,
            alternate_corr=False
        )
        
        # Load RAFT model
        self.model = torch.nn.DataParallel(RAFT(args))
        self.model.load_state_dict(torch.load(args.model, map_location=device))
        self.model = self.model.module
        self.model.to(device)
        self.model.eval()
        
        print(f"✓ RAFT model loaded from {model_path}")
    
    def compute_flow(self, img1, img2, iters=20):
        """
        Compute optical flow between two images
        
        Args:
            img1: (1, 3, H, W) torch tensor [0, 1]
            img2: (1, 3, H, W) torch tensor [0, 1]
            iters: RAFT iterations
            
        Returns:
            flow: (1, 2, H, W) optical flow [u, v]
        """
        with torch.no_grad():
            # RAFT expects [0, 255] range
            img1 = img1 * 255.0
            img2 = img2 * 255.0
            
            # Pad to multiple of 8
            _, _, h, w = img1.shape
            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8
            
            if pad_h > 0 or pad_w > 0:
                img1 = torch.nn.functional.pad(img1, [0, pad_w, 0, pad_h], mode='replicate')
                img2 = torch.nn.functional.pad(img2, [0, pad_w, 0, pad_h], mode='replicate')
            
            # Compute flow
            _, flow = self.model(img1, img2, iters=iters, test_mode=True)
            
            # Remove padding
            if pad_h > 0 or pad_w > 0:
                flow = flow[:, :, :h, :w]
            
            return flow


class SimpleFlowDetector:
    """
    Simple window detector using optical flow magnitude
    LOW flow = window opening (static background visible through window)
    HIGH flow = solid walls (moving relative to camera)
    """
    
    def __init__(self, flow_extractor, device='cuda', detection_resolution=512):
        self.flow_extractor = flow_extractor
        self.device = device
        self.detection_resolution = detection_resolution
    
    def detect_window(self, frames):
        """
        Detect window from sequence of frames
        
        Args:
            frames: List of (H, W, 3) RGB numpy arrays (0-255)
            
        Returns:
            mask: (H, W) binary mask at original resolution
            center_2d: (x, y) pixel coordinates of window center, or None
            confidence: float [0, 1]
            debug_info: dict with intermediate results
        """
        if len(frames) < 2:
            return None, None, 0.0, {}
        
        original_H, original_W = frames[0].shape[:2]
        print(f"  Original frame size: {original_W}x{original_H}")
        
        # Downsample for efficiency
        downsampled_frames = []
        for frame in frames:
            frame_small = cv2.resize(
                frame,
                (self.detection_resolution, self.detection_resolution),
                interpolation=cv2.INTER_LINEAR
            )
            downsampled_frames.append(frame_small)
        
        print(f"  Downsampled to: {self.detection_resolution}x{self.detection_resolution}")
        
        # Convert to torch tensors
        frame_0 = torch.from_numpy(downsampled_frames[0]).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        frame_last = torch.from_numpy(downsampled_frames[-1]).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        
        frame_0 = frame_0.to(self.device)
        frame_last = frame_last.to(self.device)
        
        print(f"  Computing optical flow...")
        
        # Compute optical flow
        with torch.no_grad():
            flow = self.flow_extractor.compute_flow(frame_0, frame_last)
            
            # Flow magnitude
            u = flow[0, 0]
            v = flow[0, 1]
            flow_magnitude = torch.sqrt(u**2 + v**2)
        
        Xi_np = flow_magnitude.cpu().numpy()
        
        print(f"    Flow range: [{Xi_np.min():.2f}, {Xi_np.max():.2f}]")
        print(f"    Flow mean: {Xi_np.mean():.2f}, median: {np.median(Xi_np):.2f}")
        
        # Threshold for LOW flow regions (windows)
        # Windows typically have flow < 60, walls have flow > 80
        threshold_percentile = 5  # Bottom 5% (most static regions)
        threshold = np.percentile(Xi_np, threshold_percentile)
        
        # Cap threshold to avoid selecting walls
        if threshold > 60:
            print(f"    WARNING: Threshold too high ({threshold:.2f}), capping at 60")
            threshold = 60
        
        print(f"    Threshold ({threshold_percentile}%ile): {threshold:.2f}")
        
        # Binary mask: 1 where flow < threshold
        binary_mask = (Xi_np < threshold).astype(np.uint8) * 255
        print(f"    Pixels below threshold: {np.sum(binary_mask > 0)}")
        
        # Select best bounding box
        mask_refined = self._select_best_bbox(binary_mask)
        
        # Upsample to original resolution
        mask = cv2.resize(
            mask_refined,
            (original_W, original_H),
            interpolation=cv2.INTER_NEAREST
        )
        mask = (mask > 0.5).astype(np.float32)
        
        # Compute center and confidence
        if mask.sum() > 100:
            y_coords, x_coords = np.where(mask > 0.5)
            center_x = int(x_coords.mean())
            center_y = int(y_coords.mean())
            center_2d = (center_x, center_y)
            
            mask_area = mask.sum()
            confidence = min(1.0, mask_area / (original_H * original_W * 0.3))
        else:
            center_2d = None
            confidence = 0.0
        
        debug_info = {
            'Xi': Xi_np,
            'threshold': threshold,
            'binary_mask': binary_mask,
            'mask_refined': mask_refined,
            'frames': downsampled_frames
        }
        
        print(f"  Final mask pixels: {np.sum(mask > 0.5):.0f}")
        print(f"  Center: {center_2d}, Confidence: {confidence:.3f}")
        
        return mask, center_2d, confidence, debug_info
    
    def _select_best_bbox(self, mask):
        """
        Select best bounding box from binary mask
        
        Args:
            mask: (H, W) uint8 binary mask
            
        Returns:
            result_mask: (H, W) float32 mask
        """
        H, W = mask.shape
        
        # Clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_clean, connectivity=8
        )
        
        print(f"    Found {num_labels - 1} connected components")
        
        if num_labels <= 1:
            return np.zeros((H, W), dtype=np.float32)
        
        # Evaluate components
        valid_components = []
        for label_id in range(1, num_labels):
            x, y, w, h, area = stats[label_id]
            
            # Size filters
            min_area = int(0.001 * H * W)  # At least 0.1% of image (was 0.3%)
            max_area = int(0.7 * H * W)     # At most 70% of image
            
            if area < min_area or area > max_area:
                print(f"      Component {label_id}: REJECTED (size: {area})")
                continue
            
            # Edge margin (avoid edge artifacts)
            margin = 15
            if x < margin or y < margin or x+w > W-margin or y+h > H-margin:
                print(f"      Component {label_id}: REJECTED (at edge)")
                continue
            
            # CRITICAL: Position filter - reject BOTTOM regions (likely floor)
            center_y = y + h/2
            
            # Reject bottom 60% of image (floor region)
            if center_y > H * 0.6:
                print(f"      Component {label_id}: REJECTED (bottom region - likely floor, center_y={center_y:.0f})")
                continue
            
            # Also reject very top (might be ceiling)
            if center_y < H * 0.15:
                print(f"      Component {label_id}: REJECTED (top region - likely ceiling)")
                continue
            
            # Aspect ratio filter - windows are roughly rectangular
            aspect = w / (h + 1e-6)
            if aspect < 0.4 or aspect > 4.0:  # More permissive
                print(f"      Component {label_id}: REJECTED (bad aspect: {aspect:.2f})")
                continue
            
            # Scoring: size + centrality + STRONGLY prefer MIDDLE regions (where doors/windows are)
            cx, cy = centroids[label_id]
            dist_from_center_x = abs(cx - W/2)
            max_dist_x = W/2
            horizontal_centrality = 1.0 - (dist_from_center_x / max_dist_x)
            
            # Vertical position score - PREFER middle 20-60% of image (where windows/doors typically are)
            # Create a score that peaks in middle
            vertical_norm = cy / H  # 0 = top, 1 = bottom
            if 0.2 <= vertical_norm <= 0.6:
                vertical_score = 1.0  # Perfect middle region
            elif vertical_norm < 0.2:
                vertical_score = vertical_norm / 0.2  # Penalize top
            else:  # > 0.6
                vertical_score = (1.0 - vertical_norm) / 0.4  # Penalize bottom
            
            # Combined score: size + horizontal centrality + STRONG vertical preference
            score = area * (0.3 + 0.2 * horizontal_centrality + 0.5 * vertical_score)
            
            print(f"      Component {label_id}: VALID (area={area}, cy={cy:.0f}/{H}, v_score={vertical_score:.2f}, score={score:.0f})")
            
            valid_components.append({
                'label_id': label_id,
                'score': score,
                'area': area,
                'center_y': cy
            })
        
        # Select best
        if not valid_components:
            print("    No valid components found!")
            return np.zeros((H, W), dtype=np.float32)
        
        valid_components.sort(key=lambda x: x['score'], reverse=True)
        best_label = valid_components[0]['label_id']
        print(f"    Selected component {best_label} (area={valid_components[0]['area']}, score={valid_components[0]['score']:.0f})")
        
        # Create mask
        result = (labels == best_label).astype(np.uint8) * 255
        
        # Light closing to smooth
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel_close)
        
        return result.astype(np.float32) / 255.0
    
    def visualize(self, debug_info, save_path='./log/window_detection.png'):
        """Visualize detection results"""
        import matplotlib.pyplot as plt
        
        Xi = debug_info['Xi']
        threshold = debug_info['threshold']
        binary_mask = debug_info['binary_mask']
        mask_refined = debug_info['mask_refined']
        frames = debug_info['frames']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Input frames
        axes[0, 0].imshow(frames[0])
        axes[0, 0].set_title('First Frame', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(frames[-1])
        axes[0, 1].set_title('Last Frame', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Flow magnitude
        im = axes[0, 2].imshow(Xi, cmap='jet')
        axes[0, 2].set_title('Flow Magnitude\n(BLUE = Window)', fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')
        plt.colorbar(im, ax=axes[0, 2], fraction=0.046)
        
        # Flow histogram
        axes[1, 0].hist(Xi.flatten(), bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(threshold, color='r', linestyle='--', linewidth=2,
                          label=f'Threshold: {threshold:.1f}')
        axes[1, 0].set_xlabel('Flow Magnitude')
        axes[1, 0].set_ylabel('Pixel Count')
        axes[1, 0].set_title('Flow Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Binary mask
        axes[1, 1].imshow(binary_mask, cmap='gray')
        axes[1, 1].set_title('Binary Mask (Thresholded)')
        axes[1, 1].axis('off')
        
        # Final detection
        overlay = frames[0].copy()
        mask_overlay = np.zeros_like(overlay)
        mask_overlay[mask_refined > 0.5] = [0, 255, 0]
        overlay = cv2.addWeighted(overlay, 0.7, mask_overlay, 0.3, 0)
        
        axes[1, 2].imshow(overlay)
        axes[1, 2].set_title('Final Detection')
        axes[1, 2].axis('off')
        
        plt.suptitle('Optical Flow Window Detection', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved visualization to {save_path}")
        plt.close()


class ActiveScanner:
    """
    Generates scanning trajectories for window detection
    """
    
    def __init__(self, scan_distance=0.1, num_waypoints=5):
        """
        Args:
            scan_distance: Total scanning distance (meters)
            num_waypoints: Number of waypoints in scan
        """
        self.scan_distance = scan_distance
        self.num_waypoints = num_waypoints
    
    def generate_scan_trajectory(self, start_pose):
        """
        Generate YZ diagonal scanning trajectory (in NED frame)
        
        Args:
            start_pose: dict with 'position' [x,y,z] and 'rpy' [r,p,y]
            
        Returns:
            waypoints: List of target positions [x, y, z]
        """
        x0, y0, z0 = start_pose['position']
        
        waypoints = []
        for i in range(self.num_waypoints):
            t = i / (self.num_waypoints - 1)
            offset = t * self.scan_distance
            
            # Diagonal scan in YZ plane
            # Move in Y (East) and Z (Down) simultaneously
            x = x0  # Keep X (North) constant
            y = y0 + offset / np.sqrt(2)  # East
            z = z0 + offset / np.sqrt(2)  # Down
            
            waypoints.append(np.array([x, y, z]))
        
        return waypoints


if __name__ == "__main__":
    # Test code
    print("Window detector module loaded successfully")
    print("Usage: from window_detector import OpticalFlowExtractor, SimpleFlowDetector, ActiveScanner")