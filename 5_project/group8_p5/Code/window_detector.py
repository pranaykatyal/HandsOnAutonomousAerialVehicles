"""
Window Detection using Optical Flow (from Project 4)
Integrates RAFT optical flow for TS[OK]P-style window detection
FIXED FOR SPLAT COORDINATES: Combined X+Y scanning for forward navigation
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
        
        print(f"[OK] RAFT model loaded from {model_path}")
    
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
        
        # FLOW COMPUTATION: Two options
        # 1. TS[OK]P (P4 approach): MIN across all consecutive pairs
        # 2. Standard: Just first[OK]last frame
        
        USE_TS2P = False  # [OK] CHANGE THIS TO SWITCH METHODS
        
        if USE_TS2P:
            print(f"  Computing TS[OK]P optical flow (minimum across all pairs)...")
            from scanning_fix import compute_accumulated_flow_ts2p
            Xi_np = compute_accumulated_flow_ts2p(downsampled_frames, self.flow_extractor, self.device)
        else:
            print(f"  Computing standard optical flow (first[OK]last frame)...")
            # Convert to torch tensors
            frame_first = torch.from_numpy(downsampled_frames[0]).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            frame_last = torch.from_numpy(downsampled_frames[-1]).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            frame_first = frame_first.to(self.device)
            frame_last = frame_last.to(self.device)
            
            # Compute flow
            with torch.no_grad():
                flow = self.flow_extractor.compute_flow(frame_first, frame_last)
                u = flow[0, 0]
                v = flow[0, 1]
                flow_mag = torch.sqrt(u**2 + v**2)
                Xi_np = flow_mag.cpu().numpy()
            
            print(f"    Flow first[OK]last: range [{Xi_np.min():.1f}, {Xi_np.max():.1f}], mean {Xi_np.mean():.1f}")
        
        print(f"    Flow range: [{Xi_np.min():.2f}, {Xi_np.max():.2f}]")
        print(f"    Flow mean: {Xi_np.mean():.2f}, median: {np.median(Xi_np):.2f}")
        
        # ADAPTIVE THRESHOLDING based on flow statistics
        # Window holes have LOWER flow than solid walls/edges
        # Strategy: Select flow values in the MIDDLE-LOW range
        
        # Adaptive thresholds based on flow range
        flow_min = Xi_np.min()
        flow_max = Xi_np.max()
        flow_range = flow_max - flow_min

        # Calculate percentage-based thresholds
        lower_percentage = 0.11  # 11%
        upper_percentage = 0.33  # 33%

        background_threshold = flow_min + flow_range * lower_percentage
        hole_threshold = flow_min + flow_range * upper_percentage

        print(f"    PERCENTAGE-BASED thresholding:")
        print(f"      Flow range: {flow_range:.2f} (from {flow_min:.2f} to {flow_max:.2f})")
        print(f"      Background threshold: {background_threshold:.2f} (18% of range)")
        print(f"      Hole threshold: {hole_threshold:.2f} (36% of range)")

        # Binary mask: Select flow in the percentage range
        binary_mask = ((Xi_np > background_threshold) & (Xi_np < hole_threshold)).astype(np.uint8) * 255
        
        # DEBUG before morphology
        pixels_before_morph = np.sum(binary_mask > 0)
        percentage = pixels_before_morph / binary_mask.size * 100
        print(f"    Pixels in middle flow range (before morphology): {pixels_before_morph} ({percentage:.1f}%)")
        
        # Clean up with morphology
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close)
        
        pixels_after_morph = np.sum(binary_mask > 0)
        print(f"    Pixels after morphology: {pixels_after_morph} ({pixels_after_morph/binary_mask.size*100:.1f}%)")
        
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
            'low_threshold': background_threshold,  # For visualization (lower bound)
            'high_threshold': hole_threshold,       # For visualization (upper bound)
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
            min_area = int(0.001 * H * W)  # At least 0.1% of image
            max_area = int(0.5 * H * W)     # At most 40% of image
            
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
            
            # Reject bottom 30% of image (floor region)
            if center_y > H * 0.85:
                print(f"      Component {label_id}: REJECTED (bottom region - likely floor, center_y={center_y:.0f})")
                continue
            
            # Also reject very top (might be ceiling)
            if center_y < H * 0.15:
                print(f"      Component {label_id}: REJECTED (top region - likely ceiling)")
                continue
            
            # Aspect ratio filter - windows are roughly rectangular
            aspect = w / (h + 1e-6)
            if aspect < 0.4 or aspect > 4.0:
                print(f"      Component {label_id}: REJECTED (bad aspect: {aspect:.2f})")
                continue
            
            # Scoring: size + centrality + vertical preference
            cx, cy = centroids[label_id]
            dist_from_center_x = abs(cx - W/2)
            max_dist_x = W/2
            horizontal_centrality = 1.0 - (dist_from_center_x / max_dist_x)
            
            # Vertical position score - PREFER middle 20-60%
            vertical_norm = cy / H
            if 0.2 <= vertical_norm <= 0.6:
                vertical_score = 1.0
            elif vertical_norm < 0.2:
                vertical_score = vertical_norm / 0.2
            else:
                vertical_score = (1.0 - vertical_norm) / 0.4
            
            score = area * (0.2 + 0.5 * horizontal_centrality + 0.3 * vertical_score)
            
            print(f"      Component {label_id}: VALID (area={area}, score={score:.0f})")
            
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
        
        # valid_components.sort(key=lambda x: x['score'], reverse=True)
        # valid_components.sort(key=lambda x: x['area'])
        valid_components.sort(key=lambda x: x['score'])
        best_label = valid_components[0]['label_id']
        print(f"    Selected component {best_label} (score={valid_components[0]['score']:.0f})")
        
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
        low_threshold = debug_info.get('low_threshold', None)
        high_threshold = debug_info.get('high_threshold', None)
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
        axes[0, 2].set_title('Flow Magnitude\n(GREEN = Window Frames)', fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')
        plt.colorbar(im, ax=axes[0, 2], fraction=0.046)
        
        # Flow histogram
        axes[1, 0].hist(Xi.flatten(), bins=50, alpha=0.7, edgecolor='black')
        
        # Show dual thresholds
        low_th = debug_info.get('low_threshold', None)
        high_th = debug_info.get('high_threshold', None)
        
        if low_th is not None:
            axes[1, 0].axvline(low_th, color='b', linestyle='--', linewidth=2,
                              label=f'Low: {low_th:.1f}')
        if high_th is not None:
            axes[1, 0].axvline(high_th, color='r', linestyle='--', linewidth=2,
                              label=f'High: {high_th:.1f}')
        
        axes[1, 0].set_xlabel('Flow Magnitude')
        axes[1, 0].set_ylabel('Pixel Count')
        axes[1, 0].set_title('Flow Distribution (Green = Selected Range)')
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
        print(f"  [OK] Saved visualization to {save_path}")
        plt.close()


class ActiveScanner:
    """
    Generates scanning trajectories for window detection
    [OK] FIXED: Y-only horizontal scan with increased distance for better parallax
    """
    
    def __init__(self, scan_distance=0.3, num_waypoints=5):
        """
        Args:
            scan_distance: Total scanning distance (splat units)
                          For cross pattern: d = scan_distance/4
                          scan_distance=0.3 [OK] d=0.075 per step
                          Larger steps = stronger parallax signal
            num_waypoints: Number of waypoints in scan (always 5 for cross)
        """
        self.scan_distance = scan_distance
        self.num_waypoints = num_waypoints
    
    def waypoint_to_string(self, waypoint):
        """
        Convert waypoint to string for printing (backward compatibility with tests)
        
        Args:
            waypoint: Either dict with 'position' and 'rpy' or numpy array
            
        Returns:
            String representation
        """
        if isinstance(waypoint, dict):
            pos = waypoint['position']
            yaw_deg = np.degrees(waypoint['rpy'][2])
            return f"pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], yaw={yaw_deg:+.1f}[OK]"
        elif isinstance(waypoint, np.ndarray):
            # Fallback for old array format
            return f"[{waypoint[0]:.3f}, {waypoint[1]:.3f}, {waypoint[2]:.3f}]"
        else:
            return str(waypoint)
    
    def generate_scan_trajectory(self, start_pose):
        """
        Generate CROSS/OSCILLATING scanning trajectory (P4 proven pattern)
        
        [OK] CROSS PATTERN: Oscillate around center position
        - Frame 0: Reference (center)
        - Frame 1: +Y (right)
        - Frame 2: +Z (up)
        - Frame 3: -Y (left)
        - Frame 4: -Z (down)
        
        Why this works for TS[OK]P MIN operation:
        - All frames have SIMILAR small displacement from center
        - Creates consistent parallax in all directions
        - MIN across pairs works correctly:
          * Windows (close): LOW flow in ALL directions [OK] MIN = LOW [OK]
          * Walls (far): HIGH flow in ALL directions [OK] MIN = HIGH [OK]
        - NO cumulative motion blur (each frame close to center)
        
        Args:
            start_pose: dict with 'position' [x,y,z] and 'rpy' [r,p,y]
            
        Returns:
            waypoints: List of dicts with 'position' and 'rpy'
        """
        # Handle case where start_pose might just be a position array
        if isinstance(start_pose, np.ndarray):
            x0, y0, z0 = start_pose
            roll0, pitch0, yaw0 = 0.0, 0.0, 0.0
        elif isinstance(start_pose, dict):
            x0, y0, z0 = start_pose['position']
            roll0, pitch0, yaw0 = start_pose.get('rpy', [0.0, 0.0, 0.0])
        else:
            raise ValueError(f"start_pose must be dict or array, got {type(start_pose)}")
        
        waypoints = []
        
        # Use scan_distance as the oscillation amplitude
        # For scan_distance=0.2, d=0.05 (moves [OK]0.05 from center)
        d = self.scan_distance / 4
        
        # Frame 0: Reference position (center)
        waypoints.append({
            'position': np.array([x0, y0, z0]),
            'rpy': np.array([roll0, pitch0, yaw0])
        })
        
        # Frame 1: +Y (right)
        waypoints.append({
            'position': np.array([x0, y0 + d, z0]),
            'rpy': np.array([roll0, pitch0, yaw0])
        })
        
        # Frame 2: +Z (down in NED, creates vertical parallax)
        waypoints.append({
            'position': np.array([x0, y0, z0 + d]),
            'rpy': np.array([roll0, pitch0, yaw0])
        })
        
        # Frame 3: -Y (left)
        waypoints.append({
            'position': np.array([x0, y0 - d, z0]),
            'rpy': np.array([roll0, pitch0, yaw0])
        })
        
        # Frame 4: -Z (up in NED)
        waypoints.append({
            'position': np.array([x0, y0, z0 - d]),
            'rpy': np.array([roll0, pitch0, yaw0])
        })
        
        return waypoints


if __name__ == "__main__":
    # Test code
    print("Window detector module loaded successfully")
    print("Usage: from window_detector import OpticalFlowExtractor, SimpleFlowDetector, ActiveScanner")