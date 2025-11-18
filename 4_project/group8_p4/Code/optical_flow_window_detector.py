"""
optical_flow_window_detector.py
Uses pretrained optical flow (FastFlowNet or RAFT) + lightweight window detector
More realistic and practical approach
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import cv2
from gapflytdataloader import GapFlytSequenceDataset, collate_fn


# ============================================================================
# Step 1: Optical Flow Wrapper (Use Pretrained)
# ============================================================================

class OpticalFlowExtractor:
    """
    Wrapper for pretrained optical flow models
    Support for FastFlowNet or RAFT
    """
    def __init__(self, model_type='raft', device='cuda'):
        self.device = device
        self.model_type = model_type
        
        if model_type == 'raft':
            self.model = self._load_raft()
        elif model_type == 'fastflownet':
            self.model = self._load_fastflownet()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.eval()
        print(f"Loaded pretrained {model_type.upper()} model")
    
    def _load_raft(self):
        """Load pretrained RAFT model"""
        try:
            # Try to import RAFT
            import sys
            sys.path.append('./RAFT/core')
            from raft import RAFT
            from argparse import Namespace
            
            # RAFT args
            args = Namespace(
                model='./RAFT/models/raft-things.pth',  # Pretrained weights
                small=False,
                mixed_precision=False,
                alternate_corr=False
            )
            
            model = RAFT(args)
            model.load_state_dict(torch.load(args.model))
            model = model.to(self.device)
            
            return model
        except:
            print("RAFT not found. Please clone RAFT repo:")
            print("git clone https://github.com/princeton-vl/RAFT.git")
            print("cd RAFT && ./download_models.sh")
            raise
    
    def _load_fastflownet(self):
        """Load pretrained FastFlowNet model"""
        try:
            import sys
            sys.path.append('./FastFlowNet')
            from models import FastFlowNet
            
            model = FastFlowNet()
            # Download pretrained weights from GitHub releases
            checkpoint = torch.load('./FastFlowNet/checkpoints/fastflownet_ft_sintel.pth')
            model.load_state_dict(checkpoint)
            model = model.to(self.device)
            
            return model
        except:
            print("FastFlowNet not found. Please clone FastFlowNet repo:")
            print("git clone https://github.com/ltkong218/FastFlowNet.git")
            raise
    
    @torch.no_grad()
    def compute_flow(self, frame1, frame2):
        """
        Compute optical flow between two frames
        Args:
            frame1, frame2: (B, 3, H, W) tensors in [0, 1]
        Returns:
            flow: (B, 2, H, W) - optical flow
        """
        # Normalize to model's expected input
        if self.model_type == 'raft':
            # RAFT expects [0, 255]
            frame1 = frame1 * 255.0
            frame2 = frame2 * 255.0
            
            # RAFT returns list of flow predictions (iterative refinement)
            flow_predictions = self.model(frame1, frame2, iters=20, test_mode=True)
            flow = flow_predictions[-1]  # Use final prediction
        
        elif self.model_type == 'fastflownet':
            # FastFlowNet input format
            flow = self.model(frame1, frame2)
        
        return flow
    
    @torch.no_grad()
    def extract_sequence_flows(self, frames):
        """
        Extract optical flow for entire sequence
        Args:
            frames: (B, N, 3, H, W) - sequence of frames
        Returns:
            flows: (B, N-1, 2, H, W) - flows between consecutive frames
        """
        B, N, C, H, W = frames.shape
        flows = []
        
        for i in range(N - 1):
            frame1 = frames[:, i].to(self.device)
            frame2 = frames[:, i + 1].to(self.device)
            
            flow = self.compute_flow(frame1, frame2)
            flows.append(flow)
        
        flows = torch.stack(flows, dim=1)  # (B, N-1, 2, H, W)
        return flows


# ============================================================================
# Step 2: Flow-based Feature Extraction
# ============================================================================

class FlowFeatureExtractor(nn.Module):
    """
    Extract meaningful features from optical flow for window detection
    
    Key insights:
    - Window (foreground) has minimal/different motion vs background
    - Flow magnitude differences indicate depth discontinuities
    - Flow consistency across frames indicates static structure
    """
    def __init__(self):
        super().__init__()
        
        # Flow encoding
        self.flow_encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Flow statistics encoder
        self.stats_encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),  # magnitude, angle, consistency, uncertainty
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, flows):
        """
        Args:
            flows: (B, N-1, 2, H, W) - optical flows
        Returns:
            features: (B, C, H, W) - flow features
        """
        B, N_minus_1, _, H, W = flows.shape
        
        # Compute flow statistics
        flow_magnitude = torch.norm(flows, dim=2, keepdim=True)  # (B, N-1, 1, H, W)
        flow_angle = torch.atan2(flows[:, :, 1:2], flows[:, :, 0:1])  # (B, N-1, 1, H, W)
        
        # Flow consistency across frames
        if N_minus_1 > 1:
            flow_diff = torch.diff(flows, dim=1)  # (B, N-2, 2, H, W)
            flow_consistency = 1.0 / (1.0 + torch.norm(flow_diff, dim=2).mean(dim=1, keepdim=True))  # (B, 1, H, W)
        else:
            flow_consistency = torch.ones(B, 1, H, W, device=flows.device)
        
        # Average flow magnitude
        avg_magnitude = flow_magnitude.mean(dim=1)  # (B, 1, H, W)
        avg_angle = flow_angle.mean(dim=1)  # (B, 1, H, W)
        
        # Flow uncertainty (variance across frames)
        if N_minus_1 > 1:
            flow_variance = flow_magnitude.var(dim=1, keepdim=True)  # (B, 1, H, W)
        else:
            flow_variance = torch.zeros(B, 1, H, W, device=flows.device)
        
        # Combine statistics
        flow_stats = torch.cat([avg_magnitude, avg_angle, flow_consistency, flow_variance], dim=1)  # (B, 4, H, W)
        
        # Encode statistics
        stats_features = self.stats_encoder(flow_stats)  # (B, 32, H, W)
        
        # Encode individual flows and aggregate
        flow_features_list = []
        for i in range(N_minus_1):
            flow = flows[:, i]  # (B, 2, H, W)
            flow_feat = self.flow_encoder(flow)  # (B, 128, H/8, W/8)
            flow_features_list.append(flow_feat)
        
        # Average flow features
        avg_flow_features = torch.stack(flow_features_list, dim=1).mean(dim=1)  # (B, 128, H/8, W/8)
        
        return avg_flow_features, stats_features


# ============================================================================
# Step 3: Window Detector Network
# ============================================================================

class WindowDetector(nn.Module):
    """
    Lightweight window detector using flow features
    Much smaller than TS²P since optical flow already extracts motion cues
    """
    def __init__(self):
        super().__init__()
        
        self.flow_feature_extractor = FlowFeatureExtractor()
        
        # Fusion of flow features and statistics
        self.fusion = nn.Sequential(
            nn.Conv2d(128 + 32, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # Upsample 1: 1/8 -> 1/4
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Upsample 2: 1/4 -> 1/2
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Upsample 3: 1/2 -> 1/1
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # Final prediction
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, flows):
        """
        Args:
            flows: (B, N-1, 2, H, W) - optical flows from sequence
        Returns:
            mask: (B, 1, H, W) - window segmentation
        """
        # Extract flow features
        flow_features, stats_features = self.flow_feature_extractor(flows)
        # flow_features: (B, 128, H/8, W/8)
        # stats_features: (B, 32, H, W)
        
        # Downsample stats to match flow features
        stats_features = F.interpolate(stats_features, size=flow_features.shape[-2:], mode='bilinear', align_corners=False)
        
        # Fuse features
        fused = torch.cat([flow_features, stats_features], dim=1)  # (B, 160, H/8, W/8)
        fused = self.fusion(fused)  # (B, 128, H/8, W/8)
        
        # Decode to mask
        mask = self.decoder(fused)  # (B, 1, H, W)
        
        return mask


# ============================================================================
# Step 4: Training Pipeline
# ============================================================================

def train_window_detector():
    """
    Training script for window detector with pretrained optical flow
    """
    # Hyperparameters
    batch_size = 4
    num_epochs = 30
    learning_rate = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset
    train_dataset = GapFlytSequenceDataset(
        sequences_dir="../Blender/Outputs/Sequences",
        num_frames=5,
        random_subset=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,  # Lower for flow computation
        collate_fn=collate_fn
    )
    
    # Models
    flow_extractor = OpticalFlowExtractor(model_type='raft', device=device)  # or 'fastflownet'
    window_detector = WindowDetector().to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(window_detector.parameters(), lr=learning_rate)
    
    print(f"Training Window Detector on {device}")
    print(f"Using pretrained optical flow: {flow_extractor.model_type.upper()}")
    print(f"Total sequences: {len(train_dataset)}")
    print("="*70)
    
    # Training loop
    for epoch in range(num_epochs):
        window_detector.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            frames = batch['frames']  # (B, 5, 3, 480, 640)
            masks = batch['masks']    # (B, 5, 1, 480, 640)
            
            # Use reference frame mask as target
            target_mask = masks[:, 0].to(device)  # (B, 1, 480, 640)
            
            # Extract optical flow (pretrained, no gradients)
            with torch.no_grad():
                flows = flow_extractor.extract_sequence_flows(frames)  # (B, 4, 2, 480, 640)
            
            flows = flows.to(device)
            
            # Predict window mask from flows
            pred_mask = window_detector(flows)  # (B, 1, 480, 640)
            
            # Compute loss
            loss = criterion(pred_mask, target_mask)
            
            # Backward pass (only trains window_detector, flow is frozen)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (batch_idx + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': window_detector.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'window_detector_checkpoint_epoch_{epoch+1}.pth')
    
    print("Training complete!")
    torch.save(window_detector.state_dict(), 'window_detector_final.pth')


# ============================================================================
# Step 5: Inference
# ============================================================================

@torch.no_grad()
def inference_window_detection(frames_path, checkpoint_path):
    """
    Run window detection on new sequence
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load models
    flow_extractor = OpticalFlowExtractor(model_type='raft', device=device)
    window_detector = WindowDetector().to(device)
    window_detector.load_state_dict(torch.load(checkpoint_path))
    window_detector.eval()
    
    # Load frames (implement your own loading logic)
    # frames should be (1, N, 3, H, W) tensor
    
    # Extract flows
    flows = flow_extractor.extract_sequence_flows(frames)
    
    # Detect window
    mask = window_detector(flows)
    
    return mask


if __name__ == "__main__":
    # Download pretrained optical flow models first!
    print("="*70)
    print("SETUP INSTRUCTIONS:")
    print("="*70)
    print("1. Clone RAFT (recommended):")
    print("   git clone https://github.com/princeton-vl/RAFT.git")
    print("   cd RAFT && ./download_models.sh")
    print()
    print("2. OR clone FastFlowNet:")
    print("   git clone https://github.com/ltkong218/FastFlowNet.git")
    print("   # Download pretrained weights from releases")
    print("="*70)
    
    # Uncomment to train after setup
    # train_window_detector()