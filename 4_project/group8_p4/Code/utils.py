# utils.py
"""
Visualization utilities for TS²P window detection

Provides functions to visualize optical flow, Ξ maps, and detection results.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Optional, Tuple


def visualize_optical_flow(flow: np.ndarray,
                           magnitude_threshold: float = 0.0) -> np.ndarray:
    """
    Visualize optical flow using HSV color wheel
    
    Args:
        flow: Optical flow field (H x W x 2)
        magnitude_threshold: Threshold below which flow is shown as black
        
    Returns:
        flow_rgb: RGB visualization of flow
    """
    h, w = flow.shape[:2]
    
    # Compute flow in polar coordinates
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Create HSV image
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2  # Hue = direction
    hsv[..., 1] = 255                       # Saturation = full
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value = magnitude
    
    # Threshold small flows
    hsv[..., 2][magnitude < magnitude_threshold] = 0
    
    # Convert to RGB
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return flow_rgb


def visualize_flow_magnitude(flow: np.ndarray,
                             colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Visualize flow magnitude with colormap
    
    Args:
        flow: Optical flow field (H x W x 2)
        colormap: OpenCV colormap (e.g., COLORMAP_JET, COLORMAP_VIRIDIS)
        
    Returns:
        mag_viz: RGB visualization of magnitude
    """
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    
    # Normalize to [0, 255]
    mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply colormap
    mag_viz = cv2.applyColorMap(mag_norm, colormap)
    mag_viz = cv2.cvtColor(mag_viz, cv2.COLOR_BGR2RGB)
    
    return mag_viz


def visualize_xi(xi: np.ndarray,
                 colormap: int = cv2.COLORMAP_HOT) -> np.ndarray:
    """
    Visualize Ξ edge map with colormap
    
    Args:
        xi: Ξ edge map (H x W) with values in [0, 1]
        colormap: OpenCV colormap
        
    Returns:
        xi_viz: RGB visualization
    """
    xi_norm = (xi * 255).astype(np.uint8)
    xi_viz = cv2.applyColorMap(xi_norm, colormap)
    xi_viz = cv2.cvtColor(xi_viz, cv2.COLOR_BGR2RGB)
    
    return xi_viz


def create_detection_comparison(image: np.ndarray,
                                ground_truth: Optional[np.ndarray],
                                prediction: np.ndarray) -> np.ndarray:
    """
    Create side-by-side comparison of detection vs ground truth
    
    Args:
        image: Original RGB image
        ground_truth: Ground truth mask (or None)
        prediction: Predicted window mask
        
    Returns:
        comparison: Side-by-side visualization
    """
    h, w = image.shape[:2]
    
    if ground_truth is not None:
        # Create colored comparison
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        # True positive: green
        tp = np.logical_and(ground_truth > 0, prediction > 0)
        overlay[tp] = [0, 255, 0]
        
        # False positive: red
        fp = np.logical_and(ground_truth == 0, prediction > 0)
        overlay[fp] = [255, 0, 0]
        
        # False negative: yellow
        fn = np.logical_and(ground_truth > 0, prediction == 0)
        overlay[fn] = [255, 255, 0]
        
        # Blend with original image
        vis_pred = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
        
        # Also show ground truth
        gt_overlay = np.zeros_like(image)
        gt_overlay[ground_truth > 0] = [255, 255, 255]
        vis_gt = cv2.addWeighted(image, 0.6, gt_overlay, 0.4, 0)
        
        # Side-by-side
        comparison = np.hstack([vis_gt, vis_pred])
    else:
        # Just show prediction
        overlay = np.zeros_like(image)
        overlay[prediction > 0] = [0, 255, 0]
        vis_pred = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        comparison = vis_pred
    
    return comparison


def create_full_pipeline_visualization(frames: list,
                                       flows: list,
                                       xi: np.ndarray,
                                       window_mask: np.ndarray,
                                       safe_point: Optional[Tuple[int, int]] = None) -> plt.Figure:
    """
    Create comprehensive visualization of entire TS²P pipeline
    
    Args:
        frames: List of input frames
        flows: List of optical flow fields
        xi: Computed Ξ edge map
        window_mask: Detected window mask
        safe_point: Optional safe navigation point (x, y)
        
    Returns:
        fig: Matplotlib figure with all visualizations
    """
    num_frames = len(frames)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Top row: input frames
    for i, frame in enumerate(frames[:4]):  # Show first 4 frames
        ax = plt.subplot(4, 4, i + 1)
        if len(frame.shape) == 3:
            ax.imshow(frame)
        else:
            ax.imshow(frame, cmap='gray')
        ax.set_title(f'Frame {i}')
        ax.axis('off')
    
    # Second row: optical flow visualizations
    for i, flow in enumerate(flows[:4]):
        ax = plt.subplot(4, 4, 5 + i)
        flow_vis = visualize_optical_flow(flow)
        ax.imshow(flow_vis)
        ax.set_title(f'Flow {i}')
        ax.axis('off')
    
    # Third row: flow magnitudes
    for i, flow in enumerate(flows[:4]):
        ax = plt.subplot(4, 4, 9 + i)
        mag_vis = visualize_flow_magnitude(flow)
        ax.imshow(mag_vis)
        ax.set_title(f'Magnitude {i}')
        ax.axis('off')
    
    # Bottom row: final results
    # Ξ map
    ax = plt.subplot(4, 4, 13)
    xi_vis = visualize_xi(xi)
    ax.imshow(xi_vis)
    ax.set_title('Ξ (Edge Map)')
    ax.axis('off')
    
    # Detected mask
    ax = plt.subplot(4, 4, 14)
    ax.imshow(window_mask, cmap='gray')
    ax.set_title('Detected Window')
    ax.axis('off')
    
    # Overlay on last frame
    ax = plt.subplot(4, 4, 15)
    last_frame = frames[-1]
    if len(last_frame.shape) == 2:
        last_frame = cv2.cvtColor(last_frame, cv2.COLOR_GRAY2RGB)
    
    overlay = last_frame.copy()
    overlay[window_mask > 0, 1] = 255  # Green channel
    vis = cv2.addWeighted(last_frame, 0.7, overlay, 0.3, 0)
    
    # Draw contour
    contours, _ = cv2.findContours(window_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
    
    # Mark safe point
    if safe_point is not None:
        cv2.circle(vis, safe_point, 10, (255, 0, 0), -1)
        cv2.circle(vis, safe_point, 15, (255, 0, 0), 2)
    
    ax.imshow(vis)
    ax.set_title('Final Result')
    ax.axis('off')
    
    # Statistics
    ax = plt.subplot(4, 4, 16)
    ax.axis('off')
    
    stats_text = f"""
Detection Statistics:

Frames used: {num_frames}
Window pixels: {np.sum(window_mask > 0)}
Max Ξ value: {xi.max():.3f}
Mean Ξ value: {xi.mean():.3f}
"""
    
    if safe_point is not None:
        stats_text += f"\nSafe point: ({safe_point[0]}, {safe_point[1]})"
    
    ax.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
            fontfamily='monospace')
    
    plt.tight_layout()
    
    return fig


def create_video_from_detection(frames: list,
                                masks: list,
                                output_path: str,
                                fps: int = 30,
                                show_fpv: bool = True) -> None:
    """
    Create video showing detection results
    
    Args:
        frames: List of RGB frames
        masks: List of corresponding window masks
        output_path: Output video file path
        fps: Frames per second
        show_fpv: If True, show side-by-side FPV and segmentation
    """
    import imageio
    
    video_frames = []
    
    for frame, mask in zip(frames, masks):
        if show_fpv:
            # Create overlay
            overlay = frame.copy()
            overlay[mask > 0, 1] = 255  # Green
            vis = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            # Side-by-side
            combined = np.hstack([frame, vis])
            video_frames.append(combined)
        else:
            # Just overlay
            overlay = frame.copy()
            overlay[mask > 0, 1] = 255
            vis = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            video_frames.append(vis)
    
    # Save video
    imageio.mimsave(output_path, video_frames, fps=fps)
    print(f"Video saved to {output_path}")


def plot_metrics_over_time(metrics_history: dict,
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot detection metrics over time
    
    Args:
        metrics_history: Dictionary with lists of metrics over time
            e.g., {'window_area': [...], 'xi_max': [...], ...}
        save_path: Optional path to save figure
        
    Returns:
        fig: Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for idx, (name, values) in enumerate(metrics_history.items()):
        if idx >= 4:
            break
        
        ax = axes[idx]
        ax.plot(values, linewidth=2)
        ax.set_xlabel('Frame')
        ax.set_ylabel(name)
        ax.set_title(name.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


# Example usage
if __name__ == "__main__":
    # Create dummy data for demonstration
    h, w = 480, 640
    
    # Dummy flow
    flow = np.random.randn(h, w, 2) * 5
    
    # Visualize
    flow_rgb = visualize_optical_flow(flow)
    mag_viz = visualize_flow_magnitude(flow)
    
    # Display
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(flow_rgb)
    ax1.set_title('Flow Direction')
    ax1.axis('off')
    
    ax2.imshow(mag_viz)
    ax2.set_title('Flow Magnitude')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('flow_visualization_example.png')
    print("Saved example visualization")