"""
TS²P Optical Flow Utilities
Implements the accumulated flow computation used in Project 4
"""

import torch
import numpy as np


def compute_accumulated_flow_ts2p(frames, flow_extractor, device):
    """
    Compute accumulated flow using TS²P method (MINIMUM across all pairs)
    
    This is the SECRET SAUCE that makes window detection work!
    - Computes optical flow between ALL consecutive frame pairs
    - Takes MINIMUM flow magnitude at each pixel
    - Result: Clean separation between windows (low) and walls (high)
    
    Args:
        frames: List of (H, W, 3) RGB numpy arrays (0-255)
        flow_extractor: OpticalFlowExtractor instance with compute_flow method
        device: torch device ('cuda' or 'cpu')
    
    Returns:
        Xi: (H, W) numpy array of MINIMUM flow magnitudes
    """
    if len(frames) < 2:
        raise ValueError("Need at least 2 frames for flow computation")
    
    H, W = frames[0].shape[:2]
    
    # Initialize with very high values (will be replaced by minimum)
    Xi_accumulated = np.full((H, W), float('inf'), dtype=np.float32)
    
    # Compute flow for ALL consecutive pairs
    num_pairs = len(frames) - 1
    
    for i in range(num_pairs):
        # Convert frames to torch tensors
        frame_i = torch.from_numpy(frames[i]).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        frame_next = torch.from_numpy(frames[i+1]).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        
        frame_i = frame_i.to(device)
        frame_next = frame_next.to(device)
        
        # Compute optical flow
        with torch.no_grad():
            flow = flow_extractor.compute_flow(frame_i, frame_next, iters=20)
            
            # Flow magnitude
            u = flow[0, 0]
            v = flow[0, 1]
            flow_magnitude = torch.sqrt(u**2 + v**2)
            
            # Convert to numpy
            flow_mag_np = flow_magnitude.cpu().numpy()
        
        # Take MINIMUM at each pixel (key step!)
        Xi_accumulated = np.minimum(Xi_accumulated, flow_mag_np)
    
    # Xi_accumulated now contains the MINIMUM flow across all pairs
    # This filters out noise and gives clean window detection
    
    return Xi_accumulated


def generate_scanning_positions_fixed(start_position, scan_distance=0.01, num_waypoints=5):
    """
    Generate scanning trajectory positions for active parallax
    
    Args:
        start_position: (3,) numpy array [x, y, z]
        scan_distance: Total scanning distance (meters or splat units)
        num_waypoints: Number of waypoints in scan trajectory
    
    Returns:
        positions: List of (3,) numpy arrays
    """
    positions = []
    
    # Generate diagonal scan trajectory (Y-Z plane)
    for i in range(num_waypoints):
        progress = i / (num_waypoints - 1) if num_waypoints > 1 else 0
        
        # Diagonal motion in Y-Z plane
        offset_y = progress * scan_distance / np.sqrt(2)
        offset_z = progress * scan_distance / np.sqrt(2)
        
        scan_pos = start_position.copy()
        scan_pos[1] += offset_y  # Y (East in NED)
        scan_pos[2] += offset_z  # Z (Down in NED)
        
        positions.append(scan_pos)
    
    return positions