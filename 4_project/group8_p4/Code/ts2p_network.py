"""
ts2p_gapflyt.py
Correct TS²P implementation consistent with GapFlyt paper

TS²P = Temporally Stacked Spatial Parallax
Uses optical flow to detect depth discontinuities via motion parallax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path


# =============================================================================
# Mathematical Notation (Following GapFlyt Paper)
# =============================================================================
"""
Coordinate Frames:
- ^W: World frame
- ^B: Body (quadrotor) frame  
- ^I: Image frame

Variables (using paper's notation):
- Z_F: Depth of foreground (window plane)
- Z_B: Depth of background (wall behind window)
- u, v: Optical flow components in x, y directions
- F_ij: Optical flow between frame i and frame j
- Ξ (Xi): Flow magnitude metric for gap detection
- χ (chi): Classifier output (+1 for foreground, -1 for background)
"""


class OpticalFlowExtractor:
    """
    Pretrained optical flow extractor
    GapFlyt uses FlowNet2, we can use RAFT or FastFlowNet
    """
    def __init__(self, model_type='raft', device='cuda'):
        self.device = device
        self.model_type = model_type
        
        if model_type == 'raft':
            self.model = self._load_raft()
        elif model_type == 'fastflownet':
            self.model = self._load_fastflownet()
        
        self.model.eval()
        print(f"Loaded pretrained {model_type.upper()} for optical flow")
    
    def _load_raft(self):
        """Load pretrained RAFT"""
        try:
            import sys
            sys.path.insert(0, './RAFT/core')  # Use insert instead of append
            
            from raft import RAFT
            from argparse import Namespace
            
            # RAFT args
            args = Namespace(
                model='./RAFT/models/raft-things.pth',
                small=False,
                mixed_precision=False,
                alternate_corr=False
            )
            
            model = torch.nn.DataParallel(RAFT(args))  # RAFT expects DataParallel
            model.load_state_dict(torch.load(args.model, map_location=self.device))
            model = model.module  # Unwrap from DataParallel
            model = model.to(self.device)
            
            return model
        except Exception as e:
            print(f"Error loading RAFT: {e}")
            print("Make sure RAFT is cloned: git clone https://github.com/princeton-vl/RAFT.git")
            print("And weights downloaded: cd RAFT && ./download_models.sh")
            raise
    
    def _load_fastflownet(self):
        """Load pretrained FastFlowNet"""
        try:
            import sys
            sys.path.append('./FastFlowNet')
            from models import FastFlowNet
            
            model = FastFlowNet()
            checkpoint = torch.load('./FastFlowNet/checkpoints/fastflownet_ft_sintel.pth')
            model.load_state_dict(checkpoint)
            model = model.to(self.device)
            return model
        except:
            print("FastFlowNet not found. Clone: git clone https://github.com/ltkong218/FastFlowNet.git")
            raise
    
    @torch.no_grad()
    def compute_flow(self, I_i, I_j):
        """
        Compute optical flow F_ij between frames I_i and I_j
        
        Args:
            I_i: Frame i, shape (B, 3, H, W), range [0, 1]
            I_j: Frame j, shape (B, 3, H, W), range [0, 1]
        
        Returns:
            F_ij: Optical flow from i to j, shape (B, 2, H, W)
                  F_ij[b, 0, y, x] = u (horizontal flow)
                  F_ij[b, 1, y, x] = v (vertical flow)
        """
        if self.model_type == 'raft':
            I_i = I_i * 255.0  # RAFT expects [0, 255]
            I_j = I_j * 255.0
            flow_predictions = self.model(I_i, I_j, iters=20, test_mode=True)
            F_ij = flow_predictions[-1]  # Final refined flow
        elif self.model_type == 'fastflownet':
            F_ij = self.model(I_i, I_j)
        
        return F_ij


class TS2P_GapDetector:
    """
    Temporally Stacked Spatial Parallax (TS²P) Gap Detector
    Following GapFlyt paper methodology
    
    Key Insight from Paper:
    - Foreground (window) is closer → smaller optical flow magnitude
    - Background (wall) is farther → larger optical flow magnitude
    - Active vision (diagonal scan) amplifies this parallax effect
    """
    
    def __init__(self, flow_extractor, device='cuda'):
        """
        Args:
            flow_extractor: OpticalFlowExtractor instance
            device: torch device
        """
        self.flow_extractor = flow_extractor
        self.device = device
        
        # Thresholds (tune based on your data)
        self.threshold_percentile = 83  # Median split for foreground/background
        
    def compute_temporal_flow_stack(self, frames):
        """
        Compute optical flow between consecutive frames
        
        Args:
            frames: (B, N, 3, H, W) - N frames from diagonal scanning sequence
                    frames[:, 0] = reference frame (I_0)
                    frames[:, 1:] = scanning frames (I_1, ..., I_{N-1})
        
        Returns:
            F_stack: (B, N-1, 2, H, W) - Temporal stack of optical flows
                     F_stack[:, i] = F_{0,i+1} (flow from reference to scan frame i+1)
        """
        B, N, C, H, W = frames.shape
        
        # Reference frame (taken from stationary position)
        I_ref = frames[:, 0]  # (B, 3, H, W)
        
        # Compute flows from reference to each scanning frame
        F_stack = []
        for i in range(1, N):
            I_scan = frames[:, i]  # (B, 3, H, W)
            
            # F_{0,i}: Flow from reference (frame 0) to scan frame i
            F_0i = self.flow_extractor.compute_flow(I_ref, I_scan)  # (B, 2, H, W)
            F_stack.append(F_0i)
        
        F_stack = torch.stack(F_stack, dim=1)  # (B, N-1, 2, H, W)
        return F_stack
    
    def compute_flow_magnitude(self, F_stack):
        """
        Compute flow magnitude Ξ (Xi) for gap detection
        
        Ξ(x, y) = ||F_{0,i}(x, y)||_2 averaged over all i
        
        Args:
            F_stack: (B, N-1, 2, H, W) - Temporal flow stack
        
        Returns:
            Xi: (B, 1, H, W) - Average flow magnitude
        """
        # Flow magnitude for each frame: ||F||_2 = sqrt(u^2 + v^2)
        u = F_stack[:, :, 0]  # (B, N-1, H, W)
        v = F_stack[:, :, 1]  # (B, N-1, H, W)
        
        flow_magnitude_per_frame = torch.sqrt(u**2 + v**2)  # (B, N-1, H, W)
        
        # Temporal averaging (stack spatial parallax)
        Xi = flow_magnitude_per_frame.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        
        return Xi
    
    def detect_gap_threshold(self, Xi):
        """
        Detect gap using threshold on flow magnitude
        
        Gap Detection Rule (GapFlyt):
        χ(x, y) = +1  if Ξ(x, y) < threshold  (FOREGROUND - window, closer, less flow)
        χ(x, y) = -1  if Ξ(x, y) >= threshold (BACKGROUND - wall, farther, more flow)
        
        Args:
            Xi: (B, 1, H, W) - Flow magnitude
        
        Returns:
            chi: (B, 1, H, W) - Binary mask
                 1.0 = foreground (window/gap)
                 0.0 = background
        """
        B, _, H, W = Xi.shape
        
        # Adaptive thresholding per image (use median)
        # Foreground has LOWER flow magnitude
        threshold = torch.quantile(Xi.reshape(B, -1), 
                                   q=self.threshold_percentile/100.0, 
                                   dim=1, keepdim=True)  # (B, 1)
        threshold = threshold.view(B, 1, 1, 1)  # (B, 1, 1, 1)
        
        # Binary classification
        chi = (Xi < threshold).float()  # (B, 1, H, W)
        
        return chi
    
    def morphological_refinement(self, chi):
        """
        Clean up binary mask using morphological operations
        
        Args:
            chi: (B, 1, H, W) - Binary mask
        
        Returns:
            chi_refined: (B, 1, H, W) - Cleaned mask
        """
        chi_np = chi.cpu().numpy()
        chi_refined = []
        
        for b in range(chi_np.shape[0]):
            mask = (chi_np[b, 0] * 255).astype(np.uint8)
            
            # Morphological opening (remove noise)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Morphological closing (fill holes)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            chi_refined.append(mask / 255.0)
        
        chi_refined = torch.tensor(np.array(chi_refined), 
                                    dtype=torch.float32, 
                                    device=self.device).unsqueeze(1)
        return chi_refined
    
    def detect_gap(self, frames, refine=True):
        """
        Complete TS²P gap detection pipeline
        
        Args:
            frames: (B, N, 3, H, W) - Diagonal scanning sequence
            refine: Whether to apply morphological refinement
        
        Returns:
            gap_mask: (B, 1, H, W) - Detected gap/window mask
            Xi: (B, 1, H, W) - Flow magnitude (for visualization)
        """
        # Step 1: Compute temporal flow stack
        F_stack = self.compute_temporal_flow_stack(frames)
        
        # Step 2: Compute flow magnitude Ξ
        Xi = self.compute_flow_magnitude(F_stack)
        
        # Step 3: Threshold to get binary mask χ
        chi = self.detect_gap_threshold(Xi)
        
        # Step 4: Morphological refinement (optional)
        if refine:
            chi = self.morphological_refinement(chi)
        
        return chi, Xi


# =============================================================================
# Training / Evaluation
# =============================================================================

def evaluate_ts2p():
    """
    Evaluate TS²P on your generated dataset
    No training needed - this is classical + optical flow approach!
    """
    from gapflytdataloader import GapFlytSequenceDataset, collate_fn
    from torch.utils.data import DataLoader
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    dataset = GapFlytSequenceDataset(
        sequences_dir="../Blender/Outputs/Sequences",
        num_frames=5,
        random_subset=False  # Use fixed frames for evaluation
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )
    
    # Initialize TS²P detector
    flow_extractor = OpticalFlowExtractor(model_type='raft', device=device)
    ts2p_detector = TS2P_GapDetector(flow_extractor, device=device)
    
    print("="*70)
    print("Evaluating TS²P Gap Detection (GapFlyt Method)")
    print("="*70)
    
    total_iou = 0.0
    num_samples = 0
    
    for batch_idx, batch in enumerate(dataloader):
        frames = batch['frames'].to(device)  # (B, 5, 3, 480, 640)
        gt_masks = batch['masks'][:, 0].to(device)  # (B, 1, 480, 640) - reference frame mask
        
        # Detect gap using TS²P
        pred_masks, Xi = ts2p_detector.detect_gap(frames, refine=True)
        
        # Compute IoU
        intersection = (pred_masks * gt_masks).sum(dim=[1, 2, 3])
        union = ((pred_masks + gt_masks) > 0).float().sum(dim=[1, 2, 3])
        iou = (intersection / (union + 1e-6)).mean()
        
        total_iou += iou.item()
        num_samples += 1
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Batch [{batch_idx+1}/{len(dataloader)}] IoU: {iou.item():.4f}")
        
        # Visualize first batch
        if batch_idx == 0:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            
            for i in range(min(2, frames.shape[0])):
                # Reference frame
                axes[i, 0].imshow(frames[i, 0].cpu().permute(1, 2, 0))
                axes[i, 0].set_title(f"Sample {i+1}: Reference Frame")
                axes[i, 0].axis('off')
                
                # Flow magnitude
                axes[i, 1].imshow(Xi[i, 0].cpu(), cmap='jet')
                axes[i, 1].set_title(f"Flow Magnitude Ξ")
                axes[i, 1].axis('off')
                
                # Predicted mask
                axes[i, 2].imshow(pred_masks[i, 0].cpu(), cmap='gray')
                axes[i, 2].set_title(f"Predicted Gap χ")
                axes[i, 2].axis('off')
                
                # Ground truth
                axes[i, 3].imshow(gt_masks[i, 0].cpu(), cmap='gray')
                axes[i, 3].set_title(f"Ground Truth")
                axes[i, 3].axis('off')
            
            plt.tight_layout()
            plt.savefig('ts2p_evaluation_results.png', dpi=150, bbox_inches='tight')
            print("Saved visualization to ts2p_evaluation_results.png")
            plt.close()
    
    avg_iou = total_iou / num_samples
    print("="*70)
    print(f"Average IoU: {avg_iou:.4f}")
    print("="*70)


if __name__ == "__main__":
    print("="*70)
    print("TS²P Gap Detection - GapFlyt Method")
    print("="*70)
    print("Method: Optical Flow + Temporal Stacking")
    print("NO TRAINING NEEDED - Uses pretrained optical flow!")
    print("="*70)
    
    evaluate_ts2p()