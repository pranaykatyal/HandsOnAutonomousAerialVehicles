"""
Test Script for Window Detection and Navigation System
Verify all components work before full navigation
"""

import sys
import numpy as np
import torch
from window_detector import OpticalFlowExtractor, SimpleFlowDetector, ActiveScanner
from splat_render import SplatRenderer
import os


def test_raft_model():
    """Test if RAFT model loads correctly"""
    print("\n=== Testing RAFT Model ===")
    try:
        raft_path = './RAFT/models/raft-things.pth'
        if not os.path.exists(raft_path):
            print(f"✗ RAFT model not found at {raft_path}")
            print("  Please download from: https://github.com/princeton-vl/RAFT")
            return False
        
        flow_extractor = OpticalFlowExtractor(raft_path, device='cuda')
        print("✓ RAFT model loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Error loading RAFT model: {e}")
        return False


def test_renderer():
    """Test if Gaussian splat renderer works"""
    print("\n=== Testing Renderer ===")
    try:
        config_path = "../data/P5_colmap_splat/P5_colmap/splatfacto/2025-11-17_130359/config.yml"
        json_path = "../data/render_settings/render_settings.json"
        
        if not os.path.exists(config_path):
            print(f"✗ Config not found: {config_path}")
            return False
        
        if not os.path.exists(json_path):
            print(f"✗ JSON settings not found: {json_path}")
            return False
        
        renderer = SplatRenderer(config_path, json_path)
        
        # Test render
        test_pos = np.array([0.0, 0.0, 0.0])
        test_rpy = np.array([0.0, 0.0, 0.0])
        rgb, depth, _ = renderer.render(test_pos, test_rpy)
        
        print(f"✓ Renderer works! Image shape: {rgb.shape}")
        return True
    except Exception as e:
        print(f"✗ Renderer error: {e}")
        return False


def test_collision_checker():
    """Test collision checker"""
    print("\n=== Testing Collision Checker ===")
    try:
        from collisionChecker import doesItCollide
        
        # Test at origin
        result = doesItCollide(np.array([0.0, 0.0, 0.0]))
        print(f"  Origin collision: {result}")
        
        # Test at some offset
        result = doesItCollide(np.array([1.0, 0.0, 0.0]))
        print(f"  Offset collision: {result}")
        
        print("✓ Collision checker works")
        return True
    except Exception as e:
        print(f"✗ Collision checker error: {e}")
        return False


def test_scanning_pattern():
    """Test scanning trajectory generation"""
    print("\n=== Testing Scanning Pattern ===")
    try:
        scanner = ActiveScanner(scan_distance=0.1, num_waypoints=5)
        
        start_pose = {
            'position': np.array([0.0, 0.0, 0.0]),
            'rpy': np.array([0.0, 0.0, 0.0])
        }
        
        waypoints = scanner.generate_scan_trajectory(start_pose)
        
        print(f"  Generated {len(waypoints)} waypoints:")
        for i, wp in enumerate(waypoints):
            # ✅ FIXED: Arc scanning returns dicts with 'position' and 'rpy'
            if isinstance(wp, dict):
                pos = wp['position']
                yaw_deg = np.degrees(wp['rpy'][2])
                print(f"    WP{i}: pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], yaw={yaw_deg:+.1f}°")
            else:
                # Backward compatibility for old array format
                print(f"    WP{i}: [{wp[0]:.3f}, {wp[1]:.3f}, {wp[2]:.3f}]")
        
        print("✓ Scanning pattern generation works")
        return True
    except Exception as e:
        print(f"✗ Scanning error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_optical_flow():
    """Test optical flow computation on dummy images"""
    print("\n=== Testing Optical Flow ===")
    try:
        raft_path = './RAFT/models/raft-things.pth'
        flow_extractor = OpticalFlowExtractor(raft_path, device='cuda')
        
        # Create dummy images
        img1 = torch.rand(1, 3, 512, 512).cuda()
        img2 = torch.rand(1, 3, 512, 512).cuda()
        
        flow = flow_extractor.compute_flow(img1, img2)
        
        print(f"  Flow shape: {flow.shape}")
        print(f"  Flow range: [{flow.min():.2f}, {flow.max():.2f}]")
        
        print("✓ Optical flow computation works")
        return True
    except Exception as e:
        print(f"✗ Optical flow error: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("SYSTEM INTEGRATION TEST")
    print("="*60)
    
    results = {
        'RAFT Model': test_raft_model(),
        'Renderer': test_renderer(),
        'Collision Checker': test_collision_checker(),
        'Scanning Pattern': test_scanning_pattern(),
        'Optical Flow': test_optical_flow()
    }
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name:25s}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n✓ ALL TESTS PASSED! Ready to run navigation.")
    else:
        print("\n✗ Some tests failed. Please fix issues before running navigation.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)