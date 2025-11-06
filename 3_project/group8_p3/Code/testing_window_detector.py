#!/usr/bin/env python3
"""
Standalone test script for WindowDetector
Tests detection on sample RGB image + segmentation mask
"""

import numpy as np
import cv2
import sys
from window_detector import WindowDetector


def test_on_sample_image(rgb_path, seg_path=None):
    """
    Test WindowDetector on a sample image
    
    Parameters:
    - rgb_path: Path to RGB image (e.g., 'rendered_frame_window_0.png')
    - seg_path: Path to segmentation mask (optional, will generate fake one if not provided)
    """
    
    print("="*60)
    print("WindowDetector Test")
    print("="*60)
    
    # Load RGB image
    print(f"\n1. Loading RGB image: {rgb_path}")
    rgb_image = cv2.imread(rgb_path)
    if rgb_image is None:
        print(f"ERROR: Could not load image from {rgb_path}")
        return False
    
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    h, w = rgb_image.shape[:2]
    print(f"   Image size: {w}x{h}")
    
    # Load or generate segmentation mask
    if seg_path:
        print(f"\n2. Loading segmentation mask: {seg_path}")
        seg_mask = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
        if seg_mask is None:
            print(f"ERROR: Could not load mask from {seg_path}")
            return False
    else:
        print(f"\n2. No segmentation provided - generating fake mask for testing")
        print("   (Creating a rectangular window in center of image)")
        seg_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Create a fake window (rectangle in center)
        center_x, center_y = w // 2, h // 2
        window_w, window_h = w // 4, h // 3
        
        x1 = center_x - window_w // 2
        y1 = center_y - window_h // 2
        x2 = center_x + window_w // 2
        y2 = center_y + window_h // 2
        
        cv2.rectangle(seg_mask, (x1, y1), (x2, y2), 255, -1)
        print(f"   Fake window at: ({x1}, {y1}) to ({x2}, {y2})")
    
    print(f"   Mask size: {seg_mask.shape}")
    print(f"   Mask range: {seg_mask.min()} to {seg_mask.max()}")
    print(f"   White pixels (window): {np.sum(seg_mask > 0)} / {seg_mask.size} ({np.sum(seg_mask > 0) / seg_mask.size * 100:.1f}%)")
    
    # Initialize detector
    print(f"\n3. Initializing WindowDetector")
    detector = WindowDetector(image_width=w, image_height=h)
    print(f"   Image center: {detector.img_center}")
    print(f"   Alignment threshold: {detector.alignment_threshold} pixels")
    print(f"   Close area ratio: {detector.close_area_ratio * 100:.1f}%")
    
    # Process segmentation
    print(f"\n4. Processing segmentation mask")
    detections = detector.process_segmentation(seg_mask)
    print(f"   Found {len(detections)} window(s)")
    
    if len(detections) == 0:
        print("\n❌ ERROR: No windows detected!")
        print("   Possible issues:")
        print("   - Segmentation mask is all black (no white pixels)")
        print("   - Window area too small (< min_area_threshold)")
        print("   - Check your UNet model output")
        
        # Save debug info
        cv2.imwrite('debug_mask.png', seg_mask)
        print(f"\n   Saved mask to: debug_mask.png (check if it has white pixels)")
        return False
    
    # Print detection info
    for i, det in enumerate(detections):
        print(f"\n   Window {i+1}:")
        print(f"     - Area: {det['area']:.0f} pixels ({det['area'] / (w*h) * 100:.2f}% of image)")
        print(f"     - Center: {det['center']}")
        print(f"     - Bbox: {det['bbox']}")
        print(f"     - Aspect ratio: {det['aspect_ratio']:.2f}")
    
    # Get closest window
    print(f"\n5. Selecting closest window")
    closest = detector.get_closest_window(detections)
    print(f"   Selected window with area: {closest['area']:.0f} pixels")
    
    # Calculate alignment
    print(f"\n6. Calculating alignment error")
    error_x, error_y, error_mag = detector.calculate_alignment_error(closest)
    print(f"   Error X (horizontal): {error_x:+.1f} pixels {'(right of center)' if error_x > 0 else '(left of center)'}")
    print(f"   Error Y (vertical): {error_y:+.1f} pixels {'(below center)' if error_y > 0 else '(above center)'}")
    print(f"   Total error: {error_mag:.1f} pixels")
    
    # Check status
    print(f"\n7. Checking window status")
    is_aligned = detector.is_aligned(closest)
    is_close = detector.is_close_enough(closest)
    
    print(f"   Aligned: {'✅ YES' if is_aligned else '❌ NO'} (threshold: {detector.alignment_threshold}px)")
    print(f"   Close enough: {'✅ YES' if is_close else '❌ NO'} (threshold: {detector.close_area_ratio*100:.1f}%)")
    
    if is_aligned and is_close:
        print(f"\n   🎯 READY TO FLY THROUGH!")
    elif is_aligned:
        print(f"\n   → Aligned but need to get closer")
    else:
        print(f"\n   ⟲ Need to correct alignment first")
    
    # Get approach distance
    distance_factor = detector.get_approach_distance(closest)
    print(f"\n   Approach distance factor: {distance_factor:.2f} (0.0=at window, 1.0=far away)")
    
    # Test navigation target computation
    print(f"\n8. Computing navigation target")
    current_pos = np.array([0.0, 0.0, -0.2])  # Example current position
    target = detector.compute_navigation_target(current_pos, closest, forward_distance=0.5)
    print(f"   Current position (NED): {current_pos}")
    print(f"   Target position (NED): {target}")
    print(f"   Delta: {target - current_pos}")
    
    # Visualize
    print(f"\n9. Creating visualization")
    vis_image = detector.visualize_detection(rgb_image, detections, closest)
    
    # Save outputs
    output_rgb = 'test_detection.png'
    output_mask = 'test_mask.png'
    
    cv2.imwrite(output_rgb, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(output_mask, seg_mask)
    
    print(f"\n   ✅ Saved visualization to: {output_rgb}")
    print(f"   ✅ Saved mask to: {output_mask}")
    
    print("\n" + "="*60)
    print("Test completed successfully! ✨")
    print("="*60)
    print("\nVisualization legend:")
    print("  - Green boxes: All detected windows")
    print("  - Red box: Closest/target window")
    print("  - Blue circles: Corner points")
    print("  - Yellow box: Bounding box")
    print("  - Purple crosshair: Image center")
    print("  - Purple line: Alignment error vector")
    print("\nOpen test_detection.png to see the results!")
    
    return True


def test_with_unet(rgb_path, model_path):
    """
    Test with actual UNet segmentation
    
    Parameters:
    - rgb_path: Path to RGB image
    - model_path: Path to trained UNet model (.pth file)
    """
    print("="*60)
    print("WindowDetector Test with UNet")
    print("="*60)
    
    try:
        from window_segmentation.window_segmentation import Window_Segmentaion
        from window_segmentation.network import Network
    except ImportError as e:
        print(f"ERROR: Could not import segmentation module: {e}")
        return False
    
    # Load RGB
    print(f"\n1. Loading RGB image: {rgb_path}")
    rgb_image = cv2.imread(rgb_path)
    if rgb_image is None:
        print(f"ERROR: Could not load {rgb_path}")
        return False
    
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    h, w = rgb_image.shape[:2]
    print(f"   Image size: {w}x{h}")
    
    # Load UNet
    print(f"\n2. Loading UNet model: {model_path}")
    segmentor = Window_Segmentaion(
        torch_network=Network,
        model_path=model_path,
        model_thresh=0.98,
        in_ch=3,
        out_ch=1,
        img_h=256,
        img_w=256
    )
    print("   ✓ Model loaded")
    
    # Run segmentation
    print(f"\n3. Running UNet inference...")
    seg_mask = segmentor.get_pred(rgb_image)
    seg_mask = cv2.normalize(seg_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    print(f"   ✓ Segmentation complete")
    print(f"   Mask shape: {seg_mask.shape}")
    print(f"   White pixels: {np.sum(seg_mask > 0)} ({np.sum(seg_mask > 0) / seg_mask.size * 100:.1f}%)")
    
    # Now test detector
    print(f"\n4. Testing WindowDetector on UNet output")
    
    # Initialize detector
    detector = WindowDetector(image_width=w, image_height=h)
    
    # Process
    detections = detector.process_segmentation(seg_mask)
    
    if len(detections) == 0:
        print("\n❌ No windows detected from UNet output!")
        cv2.imwrite('unet_output_mask.png', seg_mask)
        print(f"   Saved UNet output to: unet_output_mask.png")
        return False
    
    print(f"   ✓ Found {len(detections)} window(s)")
    
    closest = detector.get_closest_window(detections)
    error_x, error_y, error_mag = detector.calculate_alignment_error(closest)
    
    print(f"\n5. Results:")
    print(f"   Window area: {closest['area'] / (w*h) * 100:.2f}% of image")
    print(f"   Alignment error: {error_mag:.1f} pixels")
    print(f"   Aligned: {'✅ YES' if detector.is_aligned(closest) else '❌ NO'}")
    print(f"   Close: {'✅ YES' if detector.is_close_enough(closest) else '❌ NO'}")
    
    # Visualize
    vis_image = detector.visualize_detection(rgb_image, detections, closest)
    cv2.imwrite('unet_test_detection.png', cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite('unet_test_mask.png', seg_mask)
    
    print(f"\n   ✅ Saved to: unet_test_detection.png, unet_test_mask.png")
    
    return True


if __name__ == "__main__":
    import os
    
    print("\n" + "="*60)
    print("WindowDetector Test Suite")
    print("="*60)
    
    # Check which test to run
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  Test with pre-existing segmentation mask:")
        print("    python test_window_detector.py <rgb_image> <segmentation_mask>")
        print("\n  Test with fake mask (for debugging):")
        print("    python test_window_detector.py <rgb_image>")
        print("\n  Test with UNet:")
        print("    python test_window_detector.py <rgb_image> --unet <model_path>")
        print("\nExamples:")
        print("  python test_window_detector.py rendered_frame_window_0.png segmentation0.png")
        print("  python test_window_detector.py rendered_frame_window_0.png")
        print("  python test_window_detector.py rendered_frame_window_0.png --unet ../path/to/model.pth")
        sys.exit(1)
    
    rgb_path = sys.argv[1]
    
    # Check if image exists
    if not os.path.exists(rgb_path):
        print(f"\n❌ ERROR: RGB image not found: {rgb_path}")
        print("\nAvailable images in current directory:")
        for f in os.listdir('.'):
            if f.endswith(('.png', '.jpg', '.jpeg')):
                print(f"  - {f}")
        sys.exit(1)
    
    # UNet test
    if len(sys.argv) >= 4 and sys.argv[2] == '--unet':
        model_path = sys.argv[3]
        if not os.path.exists(model_path):
            print(f"\n❌ ERROR: Model file not found: {model_path}")
            sys.exit(1)
        
        success = test_with_unet(rgb_path, model_path)
    
    # Pre-existing mask test
    elif len(sys.argv) >= 3:
        seg_path = sys.argv[2]
        if not os.path.exists(seg_path):
            print(f"\n❌ ERROR: Segmentation mask not found: {seg_path}")
            sys.exit(1)
        
        success = test_on_sample_image(rgb_path, seg_path)
    
    # Fake mask test
    else:
        print("\nNo segmentation mask provided - will generate fake window for testing")
        success = test_on_sample_image(rgb_path, seg_path=None)
    
    sys.exit(0 if success else 1)