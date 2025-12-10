#!/usr/bin/env python3
"""
Batch Threshold Tester (Headless)

Generates visualization images for multiple threshold combinations.
Review the images to find the best threshold values.

Usage:
    python batch_threshold_test.py
    
Output:
    ./log/threshold_tests/threshold_L{low}_H{high}.png
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from window_detector import OpticalFlowExtractor, SimpleFlowDetector, ActiveScanner
from splat_render import SplatRenderer
import torch


def test_threshold_combination(flow_map, original_frame, low_thresh, high_thresh, output_path):
    """
    Generate visualization for a specific threshold combination
    
    Args:
        flow_map: (H, W) flow magnitude
        original_frame: (H, W, 3) RGB frame
        low_thresh: Lower threshold value
        high_thresh: Upper threshold value
        output_path: Where to save the image
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Flow map
    im = axes[0, 0].imshow(flow_map, cmap='jet')
    axes[0, 0].set_title('Flow Magnitude', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046)
    
    # 2. Histogram
    axes[0, 1].hist(flow_map.flatten(), bins=100, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 1].axvline(low_thresh, color='green', linestyle='--', linewidth=3, label=f'Low: {low_thresh:.1f}')
    axes[0, 1].axvline(high_thresh, color='red', linestyle='--', linewidth=3, label=f'High: {high_thresh:.1f}')
    
    ylim = axes[0, 1].get_ylim()
    axes[0, 1].fill_betweenx(ylim, low_thresh, high_thresh, alpha=0.3, color='yellow')
    
    axes[0, 1].set_xlabel('Flow Magnitude', fontsize=12)
    axes[0, 1].set_ylabel('Pixel Count', fontsize=12)
    axes[0, 1].set_title('Histogram', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Binary mask
    binary_mask = ((flow_map > low_thresh) & (flow_map < high_thresh)).astype(np.uint8) * 255
    axes[0, 2].imshow(binary_mask, cmap='gray')
    axes[0, 2].set_title('Binary Mask', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    # 4. Original frame
    axes[1, 0].imshow(original_frame)
    axes[1, 0].set_title('Original Frame', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # 5. Overlay
    if binary_mask.shape != original_frame.shape[:2]:
        binary_mask_resized = cv2.resize(binary_mask, 
                                        (original_frame.shape[1], original_frame.shape[0]))
    else:
        binary_mask_resized = binary_mask
    
    overlay = original_frame.copy()
    mask_colored = np.zeros_like(overlay)
    mask_colored[binary_mask_resized > 0] = [0, 255, 0]
    overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
    
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Overlay (Green = Selected)', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # 6. Statistics
    pixels_selected = np.sum(binary_mask > 0)
    total_pixels = binary_mask.size
    percentage = pixels_selected / total_pixels * 100
    
    flow_in_range = flow_map[(flow_map > low_thresh) & (flow_map < high_thresh)]
    if len(flow_in_range) > 0:
        flow_mean = flow_in_range.mean()
        flow_median = np.median(flow_in_range)
    else:
        flow_mean = flow_median = 0
    
    # Component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    num_components = num_labels - 1
    
    stats_text = f"""THRESHOLD: [{low_thresh:.1f}, {high_thresh:.1f}]
Range: {high_thresh - low_thresh:.1f}

SELECTION:
  Pixels: {pixels_selected:,} ({percentage:.1f}%)
  Flow Mean: {flow_mean:.2f}
  Flow Median: {flow_median:.2f}

COMPONENTS: {num_components}
"""
    
    if num_components > 0:
        areas = sorted([stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)], reverse=True)
        stats_text += f"Areas: {areas[:5]}\n"
    
    axes[1, 2].text(0.1, 0.5, stats_text, 
                   fontsize=13, family='monospace',
                   verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[1, 2].axis('off')
    axes[1, 2].set_title('Statistics', fontsize=14, fontweight='bold')
    
    plt.suptitle(f'Threshold Test: Low={low_thresh:.1f}, High={high_thresh:.1f}', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return percentage, num_components


def main():
    """Generate threshold test visualizations"""
    print("="*60)
    print("Batch Threshold Tester (Headless)")
    print("="*60)
    
    # Setup output directory
    output_dir = './log/threshold_tests'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize renderer
    config_path = "../data/P5_colmap_splat/P5_colmap/splatfacto/2025-11-17_130359/config.yml"
    json_path = "../data/render_settings/render_settings.json"
    
    print("\n1. Loading renderer...")
    renderer = SplatRenderer(config_path, json_path)
    
    # Initialize flow extractor
    print("2. Loading RAFT model...")
    from window_detector import ActiveScanner
    scanner = ActiveScanner(scan_distance=0.3, num_waypoints=5)
    raft_path = './RAFT/models/raft-things.pth'
    flow_extractor = OpticalFlowExtractor(raft_path, device='cuda')
    
    # Generate scan
    print("3. Generating scan frames...")
    start_pose = {
        'position': np.array([0.0, 0.0, 0.0]),
        'rpy': np.array([0.0, 0.0, 0.0])
    }
    
    scan_waypoints = scanner.generate_scan_trajectory(start_pose)
    
    scan_frames = []
    for i, wp in enumerate(scan_waypoints):
        print(f"   Rendering frame {i+1}/{len(scan_waypoints)}...")
        rgb, _, _ = renderer.render(wp['position'], wp['rpy'])
        scan_frames.append(rgb)
    
    # Compute flow EXACTLY like window_detector.py does
    print("4. Computing optical flow (matching main detector)...")
    
    # Downsample frames exactly like window_detector
    downsampled_frames = []
    for frame in scan_frames:
        frame_small = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_LINEAR)
        downsampled_frames.append(frame_small)
    
    # Use EXACT same computation as detect_window()
    USE_TS2P = False  # Match setting in window_detector.py line 124
    
    if USE_TS2P:
        print("   Computing TS²P optical flow...")
        from scanning_fix import compute_accumulated_flow_ts2p
        flow_map = compute_accumulated_flow_ts2p(downsampled_frames, flow_extractor, 'cuda')
    else:
        print("   Computing standard optical flow (first→last)...")
        frame_first = torch.from_numpy(downsampled_frames[0]).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        frame_last = torch.from_numpy(downsampled_frames[-1]).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        frame_first = frame_first.cuda()
        frame_last = frame_last.cuda()
        
        with torch.no_grad():
            flow = flow_extractor.compute_flow(frame_first, frame_last)
            u = flow[0, 0]
            v = flow[0, 1]
            flow_mag = torch.sqrt(u**2 + v**2)
            flow_map = flow_mag.cpu().numpy()
    
    print(f"   Flow range: [{flow_map.min():.1f}, {flow_map.max():.1f}]")
    print(f"   Flow mean: {flow_map.mean():.1f}, median: {np.median(flow_map):.1f}")
    
    # Test threshold combinations
    print("\n5. Testing threshold combinations...")
    
    # Define threshold ranges to test
    # Based on ACTUAL flow range in batch test: [0-100]
    # Testing realistic ranges for THIS flow scale
    low_thresholds = [5, 8, 10, 12, 14, 16, 18, 20]
    high_thresholds = [12, 15, 18, 20, 22, 25, 28, 30, 35, 40]
    
    results = []
    test_count = 0
    total_tests = len(low_thresholds) * len(high_thresholds)
    
    for low in low_thresholds:
        for high in high_thresholds:
            if low >= high:
                continue
            
            test_count += 1
            output_path = os.path.join(output_dir, f'threshold_L{low:.0f}_H{high:.0f}.png')
            
            print(f"   [{test_count}/{total_tests}] Testing Low={low:.0f}, High={high:.0f}...")
            
            percentage, num_components = test_threshold_combination(
                flow_map, downsampled_frames[0], low, high, output_path
            )
            
            results.append({
                'low': low,
                'high': high,
                'percentage': percentage,
                'components': num_components,
                'filename': os.path.basename(output_path)
            })
    
    # Generate summary report
    print("\n6. Generating summary report...")
    summary_path = os.path.join(output_dir, 'SUMMARY.txt')
    
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("THRESHOLD TESTING SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Flow Statistics:\n")
        f.write(f"  Range: [{flow_map.min():.1f}, {flow_map.max():.1f}]\n")
        f.write(f"  Mean: {flow_map.mean():.1f}\n")
        f.write(f"  Median: {np.median(flow_map):.1f}\n\n")
        
        f.write(f"Tested {len(results)} threshold combinations\n")
        f.write("="*60 + "\n\n")
        
        # Sort by best results (2-3 components, 5-20% selection)
        scored_results = []
        for r in results:
            score = 0
            # Good component count (2-3 is ideal)
            if r['components'] == 2:
                score += 10
            elif r['components'] == 3:
                score += 8
            elif r['components'] == 1:
                score += 5
            
            # Good percentage (8-15% is ideal)
            if 8 <= r['percentage'] <= 15:
                score += 10
            elif 5 <= r['percentage'] <= 20:
                score += 5
            
            scored_results.append((score, r))
        
        scored_results.sort(reverse=True, key=lambda x: x[0])
        
        f.write("TOP CANDIDATES (sorted by quality score):\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Rank':<6} {'Low':<6} {'High':<6} {'Pixels%':<10} {'Comps':<8} {'File':<30}\n")
        f.write("-" * 60 + "\n")
        
        for i, (score, r) in enumerate(scored_results[:15], 1):
            f.write(f"{i:<6} {r['low']:<6.0f} {r['high']:<6.0f} {r['percentage']:<10.1f} "
                   f"{r['components']:<8} {r['filename']:<30}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("\nRECOMMENDATIONS:\n")
        f.write("1. Review the top candidate images\n")
        f.write("2. Look for images where green overlay covers just window holes\n")
        f.write("3. Ideal: 2-3 components, 8-15% pixels selected\n")
        f.write("4. Use those threshold values in window_detector.py\n")
    
    print(f"\n{'='*60}")
    print("✓ COMPLETE!")
    print(f"{'='*60}")
    print(f"\nGenerated {len(results)} test images in: {output_dir}/")
    print(f"Summary report: {summary_path}")
    print("\nREVIEW:")
    print("  1. Look at the images in ./log/threshold_tests/")
    print("  2. Read SUMMARY.txt for recommendations")
    print("  3. Find the image where green overlay matches window holes")
    print("  4. Use those threshold values in window_detector.py")
    print("\nTop 3 candidates based on scoring:")
    for i, (score, r) in enumerate(scored_results[:3], 1):
        print(f"  {i}. Low={r['low']:.0f}, High={r['high']:.0f} "
              f"({r['percentage']:.1f}% selected, {r['components']} components)")


if __name__ == "__main__":
    main()