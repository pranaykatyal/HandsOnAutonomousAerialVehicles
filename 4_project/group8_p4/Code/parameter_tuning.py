"""
Parameter Tuning Script for TS²P Window Detection

This script helps you find optimal parameters by testing different combinations
and visualizing the results.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from window_detector import TS2PWindowDetector, compute_safe_point
from minimal_example import create_synthetic_scene, simulate_camera_motion


def test_parameter_combination(params, frames, ground_truth_mask):
    """
    Test a single parameter combination
    
    Args:
        params: Dictionary of parameters
        frames: List of frames
        ground_truth_mask: Ground truth window mask
        
    Returns:
        results: Dictionary with metrics
    """
    # Create detector with specified parameters
    detector = TS2PWindowDetector(
        n_frames=params['n_frames'],
        flow_threshold=params['flow_threshold'],
        min_window_area=params['min_window_area'],
        erosion_kernel_size=params['erosion_kernel_size']
    )
    
    # Process frames
    for frame in frames:
        detector.add_frame(frame)
    
    # Detect
    window_mask, xi = detector.detect()
    
    # Compute metrics
    results = {
        'params': params,
        'window_mask': window_mask,
        'xi': xi,
        'success': window_mask is not None
    }
    
    if window_mask is not None:
        # IoU
        intersection = np.logical_and(window_mask > 0, ground_truth_mask > 0)
        union = np.logical_or(window_mask > 0, ground_truth_mask > 0)
        iou = np.sum(intersection) / (np.sum(union) + 1e-6)
        
        # Other metrics
        window_pixels = np.sum(window_mask > 0)
        
        results['iou'] = iou
        results['window_pixels'] = window_pixels
    else:
        results['iou'] = 0.0
        results['window_pixels'] = 0
    
    # Ξ statistics
    if xi is not None:
        results['xi_max'] = xi.max()
        results['xi_mean'] = xi.mean()
        results['xi_std'] = xi.std()
    else:
        results['xi_max'] = 0.0
        results['xi_mean'] = 0.0
        results['xi_std'] = 0.0
    
    return results


def grid_search(frames, ground_truth_mask):
    """
    Perform grid search over parameter space
    """
    print("=" * 60)
    print("Parameter Grid Search")
    print("=" * 60)
    
    # Define parameter grid
    param_grid = {
        'n_frames': [3, 4, 5],
        'flow_threshold': [0.2, 0.3, 0.5, 0.7],
        'min_window_area': [500, 1000, 2000],
        'erosion_kernel_size': [3, 5, 7]
    }
    
    # Generate all combinations (for a smaller subset)
    test_configs = []
    
    # Test key combinations
    for n_frames in param_grid['n_frames']:
        for flow_threshold in param_grid['flow_threshold']:
            for min_area in [1000]:  # Fix min_area for now
                for erosion in [5]:  # Fix erosion for now
                    test_configs.append({
                        'n_frames': n_frames,
                        'flow_threshold': flow_threshold,
                        'min_window_area': min_area,
                        'erosion_kernel_size': erosion
                    })
    
    print(f"Testing {len(test_configs)} parameter combinations...\n")
    
    results = []
    for i, params in enumerate(test_configs):
        result = test_parameter_combination(params, frames, ground_truth_mask)
        results.append(result)
        
        if result['success']:
            print(f"[{i+1}/{len(test_configs)}] ✓ n={params['n_frames']}, "
                  f"τ={params['flow_threshold']:.2f} → "
                  f"IoU={result['iou']:.3f}, pixels={result['window_pixels']}")
        else:
            print(f"[{i+1}/{len(test_configs)}] ✗ n={params['n_frames']}, "
                  f"τ={params['flow_threshold']:.2f} → FAILED")
    
    # Find best result
    successful_results = [r for r in results if r['success']]
    
    if successful_results:
        best_result = max(successful_results, key=lambda x: x['iou'])
        print("\n" + "=" * 60)
        print("Best Parameters:")
        print("=" * 60)
        print(f"n_frames: {best_result['params']['n_frames']}")
        print(f"flow_threshold: {best_result['params']['flow_threshold']}")
        print(f"min_window_area: {best_result['params']['min_window_area']}")
        print(f"erosion_kernel_size: {best_result['params']['erosion_kernel_size']}")
        print(f"\nBest IoU: {best_result['iou']:.3f}")
        print(f"Window pixels: {best_result['window_pixels']}")
    else:
        print("\n✗ No successful detections found!")
        best_result = None
    
    return results, best_result


def visualize_threshold_sensitivity(frames, ground_truth_mask):
    """
    Visualize how threshold affects detection
    """
    print("\n" + "=" * 60)
    print("Threshold Sensitivity Analysis")
    print("=" * 60)
    
    # Generate frames
    detector = TS2PWindowDetector(n_frames=4, flow_threshold=0.5)
    for frame in frames:
        detector.add_frame(frame)
    
    _, xi = detector.detect()
    
    if xi is None:
        print("Failed to compute Ξ!")
        return
    
    # Test different thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, thresh in enumerate(thresholds):
        # Threshold Ξ
        threshold_val = int(thresh * 255)
        _, binary = cv2.threshold(
            (xi * 255).astype(np.uint8),
            threshold_val,
            255,
            cv2.THRESH_BINARY
        )
        
        # Count pixels
        num_pixels = np.sum(binary > 0)
        
        # Visualize
        axes[idx].imshow(binary, cmap='gray')
        axes[idx].set_title(f'τ={thresh:.1f}\n{num_pixels} pixels')
        axes[idx].axis('off')
    
    plt.suptitle('Threshold Sensitivity (Binary Ξ)', fontsize=14)
    plt.tight_layout()
    plt.savefig('threshold_sensitivity.png', dpi=150, bbox_inches='tight')
    print("✓ Saved threshold_sensitivity.png")
    
    plt.show()


def visualize_detection_pipeline(frames, ground_truth_mask, params):
    """
    Visualize complete detection pipeline with specific parameters
    """
    print(f"\n" + "=" * 60)
    print(f"Pipeline Visualization with Parameters:")
    print(f"  n_frames={params['n_frames']}, flow_threshold={params['flow_threshold']}")
    print("=" * 60)
    
    detector = TS2PWindowDetector(**params)
    
    for frame in frames:
        detector.add_frame(frame)
    
    window_mask, xi = detector.detect()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Reference frame
    axes[0, 0].imshow(frames[0], cmap='gray')
    axes[0, 0].set_title('Reference Frame')
    axes[0, 0].axis('off')
    
    # Last frame
    axes[0, 1].imshow(frames[-1], cmap='gray')
    axes[0, 1].set_title('Last Frame')
    axes[0, 1].axis('off')
    
    # Ξ map
    if xi is not None:
        xi_viz = (xi * 255).astype(np.uint8)
        axes[0, 2].imshow(xi_viz, cmap='jet')
        axes[0, 2].set_title(f'Ξ (max={xi.max():.3f})')
        axes[0, 2].axis('off')
        
        # Binary threshold
        threshold_val = int(params['flow_threshold'] * 255)
        _, binary = cv2.threshold(xi_viz, threshold_val, 255, cv2.THRESH_BINARY)
        axes[1, 0].imshow(binary, cmap='gray')
        axes[1, 0].set_title(f'Binary (τ={params["flow_threshold"]})\n{np.sum(binary>0)} pixels')
        axes[1, 0].axis('off')
    else:
        axes[0, 2].text(0.5, 0.5, 'Ξ computation failed', ha='center', va='center')
        axes[0, 2].axis('off')
        axes[1, 0].axis('off')
    
    # Ground truth
    axes[1, 1].imshow(ground_truth_mask, cmap='gray')
    axes[1, 1].set_title('Ground Truth')
    axes[1, 1].axis('off')
    
    # Detected window
    if window_mask is not None:
        axes[1, 2].imshow(window_mask, cmap='gray')
        iou = np.sum(np.logical_and(window_mask > 0, ground_truth_mask > 0)) / \
              np.sum(np.logical_or(window_mask > 0, ground_truth_mask > 0))
        axes[1, 2].set_title(f'Detected (IoU={iou:.3f})')
    else:
        axes[1, 2].text(0.5, 0.5, 'Detection Failed', ha='center', va='center')
        axes[1, 2].set_title('Detected Window')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('pipeline_visualization.png', dpi=150, bbox_inches='tight')
    print("✓ Saved pipeline_visualization.png")
    
    plt.show()


def diagnose_detection_failure(frames, ground_truth_mask):
    """
    Comprehensive diagnosis of why detection is failing
    """
    print("\n" + "=" * 60)
    print("Detection Failure Diagnosis")
    print("=" * 60)
    
    # Test with very permissive parameters
    detector = TS2PWindowDetector(
        n_frames=4,
        flow_threshold=0.1,  # Very low threshold
        min_window_area=100,  # Very small area
        erosion_kernel_size=1  # Minimal erosion
    )
    
    for frame in frames:
        detector.add_frame(frame)
    
    window_mask, xi = detector.detect()
    
    print("\n1. Ξ Statistics:")
    if xi is not None:
        print(f"   Min: {xi.min():.4f}")
        print(f"   Max: {xi.max():.4f}")
        print(f"   Mean: {xi.mean():.4f}")
        print(f"   Std: {xi.std():.4f}")
        
        # Check if Ξ has any meaningful values
        if xi.max() < 0.1:
            print("   ⚠ WARNING: Ξ values are very low!")
            print("   → This suggests insufficient optical flow or motion parallax")
        elif xi.max() > 0.5:
            print("   ✓ Ξ has good dynamic range")
    else:
        print("   ✗ Ξ computation failed!")
    
    print("\n2. Optical Flow Check:")
    # Check flow magnitudes
    flow_mags = []
    for i in range(1, len(detector.frame_buffer)):
        if detector.frame_buffer[i].flow is not None:
            mag = np.sqrt(detector.frame_buffer[i].flow[..., 0]**2 + 
                         detector.frame_buffer[i].flow[..., 1]**2)
            flow_mags.append(mag.mean())
    
    if flow_mags:
        print(f"   Average flow magnitude: {np.mean(flow_mags):.2f} pixels")
        if np.mean(flow_mags) < 1.0:
            print("   ⚠ WARNING: Flow magnitude is very small!")
            print("   → Try increasing scan_distance or motion amount")
        else:
            print("   ✓ Flow magnitude looks reasonable")
    
    print("\n3. Binary Threshold Test:")
    if xi is not None:
        for thresh in [0.1, 0.2, 0.3, 0.5]:
            _, binary = cv2.threshold(
                (xi * 255).astype(np.uint8),
                int(thresh * 255),
                255,
                cv2.THRESH_BINARY
            )
            num_pixels = np.sum(binary > 0)
            print(f"   τ={thresh:.1f}: {num_pixels:6d} pixels")
    
    print("\n4. Contour Detection Test:")
    if window_mask is not None:
        print(f"   ✓ Window detected with {np.sum(window_mask > 0)} pixels")
    else:
        print("   ✗ No valid contours found")
        print("   → Try lowering flow_threshold or min_window_area")
    
    print("\n" + "=" * 60)


def main():
    """
    Main tuning script
    """
    print("=" * 60)
    print("TS²P Parameter Tuning Script")
    print("=" * 60)
    
    # Create synthetic scene
    print("\n[1] Creating synthetic scene...")
    foreground, background, ground_truth_mask = create_synthetic_scene(
        width=640, height=480, window_size=200
    )
    
    # Generate frames with motion
    print("[2] Generating frames with motion...")
    frames = []
    
    reference_frame = simulate_camera_motion(
        foreground, background, ground_truth_mask, 0, 0
    )
    frames.append(reference_frame)
    
    # Diagonal motion
    motions = [(10, 10), (20, 20), (30, 30), (40, 40)]
    for mx, my in motions:
        frame = simulate_camera_motion(
            foreground, background, ground_truth_mask, mx, my
        )
        frames.append(frame)
    
    # Run diagnosis
    print("\n" + "=" * 60)
    print("Starting Diagnostics...")
    print("=" * 60)
    
    diagnose_detection_failure(frames, ground_truth_mask)
    
    # Visualize threshold sensitivity
    visualize_threshold_sensitivity(frames, ground_truth_mask)
    
    # Grid search
    results, best_result = grid_search(frames, ground_truth_mask)
    
    # Visualize best result
    if best_result is not None:
        visualize_detection_pipeline(frames, ground_truth_mask, best_result['params'])
    else:
        print("\n⚠ No successful detection - showing with default parameters")
        default_params = {
            'n_frames': 4,
            'flow_threshold': 0.3,
            'min_window_area': 1000,
            'erosion_kernel_size': 5
        }
        visualize_detection_pipeline(frames, ground_truth_mask, default_params)
    
    print("\n" + "=" * 60)
    print("Tuning Complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - threshold_sensitivity.png")
    print("  - pipeline_visualization.png")


if __name__ == "__main__":
    main()