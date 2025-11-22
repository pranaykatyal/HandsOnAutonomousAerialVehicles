"""
Fixed scanning pattern and flow computation for TSÂ²P
"""

import numpy as np
import torch


def generate_scanning_positions_fixed(current_position, scan_distance=0.05):
    """
    Generate oscillating scanning trajectory for TSÂ²P parallax
    
    Motion pattern creates parallax by moving in 4 different directions:
    - Frame 0: Reference position
    - Frame 1: +Y (right)
    - Frame 2: +Z (up)  
    - Frame 3: -Y (left)
    - Frame 4: -Z (down)
    
    This oscillating pattern creates differential motion parallax between
    foreground (windows) and background (walls).
    
    Args:
        current_position: [x, y, z] in NED coordinates
        scan_distance: Distance to move in each direction (splat units)
    
    Returns:
        List of 5 positions
    """
    positions = []
    
    # Frame 0: Reference position
    positions.append(current_position.copy())
    
    # Frame 1: Move +Y (right)
    pos1 = current_position.copy()
    pos1[1] += scan_distance
    positions.append(pos1)
    
    # Frame 2: Move +Z (up)
    pos2 = current_position.copy()
    pos2[2] += scan_distance
    positions.append(pos2)
    
    # Frame 3: Move -Y (left)
    pos3 = current_position.copy()
    pos3[1] -= scan_distance
    positions.append(pos3)
    
    # Frame 4: Move -Z (down)
    pos4 = current_position.copy()
    pos4[2] -= scan_distance
    positions.append(pos4)
    
    return positions


def compute_accumulated_flow_ts2p(frames, flow_extractor, device='cuda'):
    """
    Compute temporal flow for TSÂ²P detection using MINIMUM
    
    Strategy: Take the MINIMUM flow magnitude across all consecutive pairs.
    
    Why MIN works best:
    - Windows (close): LOW flow in ALL directions â†’ MIN is low
    - Walls (far): HIGH flow in ALL directions â†’ MIN is still high
    - Edges/artifacts: Might have low flow in ONE direction but high in others â†’ MIN catches this
    
    This is more robust than averaging or max.
    
    Args:
        frames: List of 5 RGB numpy arrays (H, W, 3)
        flow_extractor: OpticalFlowExtractor instance
        device: torch device
        
    Returns:
        Xi_min: (H, W) minimum flow magnitude across all pairs
    """
    # Convert frames to torch
    frames_tensor = []
    for frame in frames:
        frame_t = torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        frames_tensor.append(frame_t.to(device))
    
    # Compute flow between consecutive frames
    flow_magnitudes = []
    
    with torch.no_grad():
        for i in range(len(frames_tensor) - 1):
            # Flow from frame i to frame i+1
            flow = flow_extractor.compute_flow(frames_tensor[i], frames_tensor[i+1])
            
            # Flow magnitude
            u = flow[0, 0]
            v = flow[0, 1]
            flow_mag = torch.sqrt(u**2 + v**2)
            
            flow_magnitudes.append(flow_mag)
            # Removed verbose flow print
    
    # Stack and take MINIMUM
    flow_stack = torch.stack(flow_magnitudes, dim=0)
    Xi_min = torch.min(flow_stack, dim=0)[0]  # Min across temporal dimension
    
    return Xi_min.cpu().numpy()


# Usage in main.py:
"""
# Replace generate_scanning_positions with:
from scanning_fix import generate_scanning_positions_fixed

scan_positions = generate_scanning_positions_fixed(currentPose['position'], 
                                                    scan_distance=0.05)

# In improved detector, replace the flow computation with:
from scanning_fix import compute_accumulated_flow_ts2p

Xi_np = compute_accumulated_flow_ts2p(downsampled_frames, 
                                      self.flow_extractor, 
                                      self.device)
"""