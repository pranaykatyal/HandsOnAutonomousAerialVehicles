"""
Debug script to see exactly where detection is failing
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from window_detector import TS2PWindowDetector


def create_simple_test_scene(width=640, height=480, window_size=200):
    """
    Create the SIMPLEST possible test scene
    """
    # Very bright background
    background = np.ones((height, width), dtype=np.uint8) * 200
    
    # Very dark foreground
    foreground = np.ones((height, width), dtype=np.uint8) * 50
    
    # Window in center
    center_x, center_y = width // 2, height // 2
    window_x1 = center_x - window_size // 2
    window_y1 = center_y - window_size // 2
    window_x2 = center_x + window_size // 2
    window_y2 = center_y + window_size // 2
    
    window_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(window_mask, (window_x1, window_y1), (window_x2, window_y2), 255, -1)
    
    return foreground, background, window_mask


def simulate_simple_motion(foreground, background, window_mask, motion_x, motion_y):
    """
    Very simple motion simulation
    """
    h, w = foreground.shape
    
    # Background barely moves (10%)
    M_bg = np.float32([[1, 0, int(motion_x * 0.1)], [0, 1, int(motion_y * 0.1)]])
    
    # Foreground moves fully
    M_fg = np.float32([[1, 0, motion_x], [0, 1, motion_y]])
    
    # Apply transforms
    bg_moved = cv2.warpAffine(background, M_bg, (w, h), borderValue=200)
    fg_moved = cv2.warpAffine(foreground, M_fg, (w, h), borderValue=50)
    win_moved = cv2.warpAffine(window_mask, M_fg, (w, h))
    
    # Composite
    composite = fg_moved.copy()
    composite[win_moved > 128] = bg_moved[win_moved > 128]
    
    return composite


def debug_detection():
    """
    Debug the detection pipeline step by step
    """
    print("=" * 60)
    print("Debug Detection Pipeline")
    print("=" * 60)
    
    # Create simple scene
    print("\n[1] Creating simple scene...")
    fg, bg, gt = create_simple_test_scene()
    
    # Generate frames with LARGE motion
    print("[2] Generating frames with large motion...")
    frames = []
    frames.append(simulate_simple_motion(fg, bg, gt, 0, 0))
    
    # Use VERY large motion
    for motion in [20, 40, 60, 80]:
        frames.append(simulate_simple_motion(fg, bg, gt, motion, motion))
    
    print(f"    Generated {len(frames)} frames")
    
    # Create detector with VERY permissive settings
    print("\n[3] Creating detector with permissive settings...")
    detector = TS2PWindowDetector(
        n_frames=4,
        flow_threshold=0.05,  # VERY low
        min_window_area=100,   # VERY small
        erosion_kernel_size=1  # Minimal
    )
    
    # Add frames and check flow
    print("\n[4] Processing frames and checking optical flow...")
    for i, frame in enumerate(frames):
        ready = detector.add_frame(frame)
        print(f"    Frame {i}: added, ready={ready}")
        
        if i > 0 and len(detector.frame_buffer) > 1:
            flow = detector.frame_buffer[i].flow
            if flow is not None:
                flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                print(f"      Flow magnitude: min={flow_mag.min():.2f}, "
                      f"max={flow_mag.max():.2f}, mean={flow_mag.mean():.2f}")
    
    # Try to detect
    print("\n[5] Attempting detection...")
    try:
        window_mask, xi = detector.detect()
        
        if xi is None:
            print("    ✗ Ξ computation returned None")
            return
        
        print(f"    ✓ Ξ computed: min={xi.min():.3f}, max={xi.max():.3f}, mean={xi.mean():.3f}")
        
        # Check binary threshold
        threshold_val = int(detector.flow_threshold * 255)
        _, binary = cv2.threshold((xi * 255).astype(np.uint8), threshold_val, 255, cv2.THRESH_BINARY)
        print(f"    Binary pixels at τ={detector.flow_threshold}: {np.sum(binary > 0)}")
        
        if window_mask is None:
            print("    ✗ Window mask is None (contour detection failed)")
        elif np.sum(window_mask) == 0:
            print("    ✗ Window mask is empty (no pixels)")
        else:
            print(f"    ✓ Window detected: {np.sum(window_mask > 0)} pixels")
            
            # Compute IoU
            intersection = np.logical_and(window_mask > 0, gt > 0)
            union = np.logical_or(window_mask > 0, gt > 0)
            iou = np.sum(intersection) / np.sum(union)
            print(f"    IoU with ground truth: {iou:.3f}")
        
    except Exception as e:
        print(f"    ✗ Exception during detection: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Visualize
    print("\n[6] Creating visualization...")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Frames
    axes[0, 0].imshow(frames[0], cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title('Frame 0 (Reference)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(frames[-1], cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title(f'Frame {len(frames)-1} (Last)')
    axes[0, 1].axis('off')
    
    # Optical flow
    if detector.frame_buffer[1].flow is not None:
        flow = detector.frame_buffer[1].flow
        flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        im = axes[0, 2].imshow(flow_mag, cmap='jet')
        axes[0, 2].set_title(f'Flow Magnitude\nmax={flow_mag.max():.1f}')
        axes[0, 2].axis('off')
        plt.colorbar(im, ax=axes[0, 2])
    
    # Ξ
    if xi is not None:
        im = axes[0, 3].imshow(xi, cmap='jet', vmin=0, vmax=1)
        axes[0, 3].set_title(f'Ξ (Edge Map)\nmax={xi.max():.3f}')
        axes[0, 3].axis('off')
        plt.colorbar(im, ax=axes[0, 3])
        
        # Binary threshold
        axes[1, 0].imshow(binary, cmap='gray')
        axes[1, 0].set_title(f'Binary (τ={detector.flow_threshold})\n{np.sum(binary>0)} pixels')
        axes[1, 0].axis('off')
    
    # Ground truth
    axes[1, 1].imshow(gt, cmap='gray')
    axes[1, 1].set_title('Ground Truth')
    axes[1, 1].axis('off')
    
    # Detection result
    if window_mask is not None and np.sum(window_mask) > 0:
        axes[1, 2].imshow(window_mask, cmap='gray')
        axes[1, 2].set_title(f'Detected\n{np.sum(window_mask>0)} pixels')
    else:
        axes[1, 2].text(0.5, 0.5, 'No Detection', ha='center', va='center')
        axes[1, 2].set_title('Detected Window')
    axes[1, 2].axis('off')
    
    # Overlay
    if window_mask is not None and np.sum(window_mask) > 0:
        overlay = np.zeros((frames[0].shape[0], frames[0].shape[1], 3), dtype=np.uint8)
        overlay[..., 0] = frames[-1]
        overlay[..., 1] = frames[-1]
        overlay[..., 2] = frames[-1]
        
        # Green = TP, Red = FP, Yellow = FN
        tp = np.logical_and(window_mask > 0, gt > 0)
        fp = np.logical_and(window_mask > 0, gt == 0)
        fn = np.logical_and(window_mask == 0, gt > 0)
        
        overlay[tp] = [0, 255, 0]
        overlay[fp] = [255, 0, 0]
        overlay[fn] = [255, 255, 0]
        
        axes[1, 3].imshow(overlay)
        axes[1, 3].set_title('Overlay\nG=TP, R=FP, Y=FN')
    else:
        axes[1, 3].axis('off')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('debug_detection.png', dpi=150, bbox_inches='tight')
    print("✓ Saved debug_detection.png")
    
    plt.show()
    
    print("\n" + "=" * 60)
    print("Debug complete!")
    print("=" * 60)


if __name__ == "__main__":
    debug_detection()