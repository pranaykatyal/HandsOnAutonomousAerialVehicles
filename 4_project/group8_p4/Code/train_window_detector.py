"""
train_window_detector.py
Option B: Trainable window detector on top of pretrained optical flow
Handles edge cases that thresholding misses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from ts2p_network import OpticalFlowExtractor
from gapflytdataloader import GapFlytSequenceDataset, collate_fn


# =============================================================================
# Helper Function for Flow Extraction
# =============================================================================

@torch.no_grad()
def extract_sequence_flows(flow_extractor, frames):
    """
    Extract optical flow for entire sequence
    Args:
        flow_extractor: OpticalFlowExtractor instance
        frames: (B, N, 3, H, W) - sequence of frames
    Returns:
        flows: (B, N-1, 2, H, W) - flows between consecutive frames
    """
    B, N, C, H, W = frames.shape
    flows = []
    
    # Reference frame
    I_ref = frames[:, 0].to(flow_extractor.device)
    
    for i in range(1, N):
        I_scan = frames[:, i].to(flow_extractor.device)
        
        # Compute flow from reference to scan frame
        flow = flow_extractor.compute_flow(I_ref, I_scan)
        flows.append(flow)
    
    flows = torch.stack(flows, dim=1)  # (B, N-1, 2, H, W)
    return flows


# =============================================================================
# Flow Feature Extractor (Enhanced)
# =============================================================================

class FlowFeatureExtractor(nn.Module):
    """
    Extract rich features from optical flow for window detection
    
    Features extracted:
    - u, v: Raw flow components
    - ||F||: Flow magnitude
    - θ: Flow angle
    - σ_temporal: Temporal variance across frames
    - ∇||F||: Flow magnitude gradients (edges)
    """
    
    def __init__(self):
        super().__init__()
        
        # Learnable feature encoder
        self.flow_encoder = nn.Sequential(
            # Input: (u, v) per frame → 2 * (N-1) channels
            nn.Conv2d(8, 32, kernel_size=7, padding=3),  # 2*(N-1)=8 for N=5
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Statistics encoder (hand-crafted features)
        self.stats_encoder = nn.Sequential(
            # magnitude_mean, magnitude_var, angle_mean, angle_var, edge_strength
            nn.Conv2d(5, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
    def compute_flow_statistics(self, F_stack):
        """
        Compute statistical features from flow stack
        
        Args:
            F_stack: (B, N-1, 2, H, W) - optical flows
        Returns:
            stats: (B, 5, H, W) - statistical features
        """
        B, N_minus_1, _, H, W = F_stack.shape
        
        # Flow magnitude: ||F|| = sqrt(u^2 + v^2)
        u = F_stack[:, :, 0]  # (B, N-1, H, W)
        v = F_stack[:, :, 1]  # (B, N-1, H, W)
        magnitude = torch.sqrt(u**2 + v**2)  # (B, N-1, H, W)
        
        # Flow angle: θ = atan2(v, u)
        angle = torch.atan2(v, u)  # (B, N-1, H, W)
        
        # Temporal statistics
        magnitude_mean = magnitude.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        magnitude_var = magnitude.var(dim=1, keepdim=True)    # (B, 1, H, W)
        angle_mean = angle.mean(dim=1, keepdim=True)          # (B, 1, H, W)
        angle_var = angle.var(dim=1, keepdim=True)            # (B, 1, H, W)
        
        # Edge strength (gradient of magnitude)
        # Sobel-like edge detection on mean magnitude
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                dtype=torch.float32, device=F_stack.device)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                dtype=torch.float32, device=F_stack.device)
        kernel_x = kernel_x.view(1, 1, 3, 3)
        kernel_y = kernel_y.view(1, 1, 3, 3)
        
        grad_x = F.conv2d(magnitude_mean, kernel_x, padding=1)
        grad_y = F.conv2d(magnitude_mean, kernel_y, padding=1)
        edge_strength = torch.sqrt(grad_x**2 + grad_y**2)  # (B, 1, H, W)
        
        # Stack all statistics
        stats = torch.cat([
            magnitude_mean,
            magnitude_var,
            angle_mean,
            angle_var,
            edge_strength
        ], dim=1)  # (B, 5, H, W)
        
        return stats
    
    def forward(self, F_stack):
        """
        Args:
            F_stack: (B, N-1, 2, H, W)
        Returns:
            features: (B, 96, H, W) - combined features
        """
        B, N_minus_1, _, H, W = F_stack.shape
        
        # Flatten temporal dimension for raw flow encoding
        F_flat = F_stack.view(B, N_minus_1 * 2, H, W)  # (B, 2*(N-1), H, W)
        
        # Encode raw flows
        flow_features = self.flow_encoder(F_flat)  # (B, 64, H, W)
        
        # Compute and encode statistics
        stats = self.compute_flow_statistics(F_stack)  # (B, 5, H, W)
        stats_features = self.stats_encoder(stats)      # (B, 32, H, W)
        
        # Concatenate
        features = torch.cat([flow_features, stats_features], dim=1)  # (B, 96, H, W)
        
        return features


# =============================================================================
# Window Detector Network
# =============================================================================

class WindowDetectorNetwork(nn.Module):
    """
    Lightweight U-Net style network for window segmentation
    Input: Flow features (B, 96, H, W)
    Output: Window mask (B, 1, H, W)
    """
    
    def __init__(self, in_channels=96):
        super().__init__()
        
        # Encoder
        self.enc1 = self._make_encoder_block(in_channels, 64)
        self.enc2 = self._make_encoder_block(64, 128)
        self.enc3 = self._make_encoder_block(128, 256)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Decoder with skip connections
        self.dec3 = self._make_decoder_block(512 + 256, 256)
        self.dec2 = self._make_decoder_block(256 + 128, 128)
        self.dec1 = self._make_decoder_block(128 + 64, 64)
        
        # Final prediction
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def _make_encoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def _make_decoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        # Encoder with skip connections
        e1 = self.enc1(x)           # (B, 64, H, W)
        e2 = self.enc2(F.max_pool2d(e1, 2))  # (B, 128, H/2, W/2)
        e3 = self.enc3(F.max_pool2d(e2, 2))  # (B, 256, H/4, W/4)
        
        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e3, 2))  # (B, 512, H/8, W/8)
        
        # Decoder with skip connections
        d3 = F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e3], dim=1)  # Skip connection
        d3 = self.dec3(d3)  # (B, 256, H/4, W/4)
        
        d2 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)  # (B, 128, H/2, W/2)
        
        d1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)  # (B, 64, H, W)
        
        # Final prediction
        mask = self.final(d1)  # (B, 1, H, W)
        
        return mask


# =============================================================================
# Combined Detector
# =============================================================================

class TrainableWindowDetector(nn.Module):
    """
    Complete trainable window detector
    Combines flow feature extraction + segmentation network
    """
    
    def __init__(self):
        super().__init__()
        self.feature_extractor = FlowFeatureExtractor()
        self.segmentation_net = WindowDetectorNetwork(in_channels=96)
        
    def forward(self, F_stack):
        """
        Args:
            F_stack: (B, N-1, 2, H, W) - optical flows
        Returns:
            mask: (B, 1, H, W) - window segmentation
        """
        features = self.feature_extractor(F_stack)  # (B, 96, H, W)
        mask = self.segmentation_net(features)       # (B, 1, H, W)
        return mask


# =============================================================================
# Loss Functions
# =============================================================================

class DiceBCELoss(nn.Module):
    """
    Combined Dice + BCE loss for better boundary detection
    """
    
    def __init__(self, weight_bce=0.5, weight_dice=0.5):
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.bce = nn.BCELoss()
        
    def dice_loss(self, pred, target, smooth=1e-6):
        """Soft Dice loss"""
        intersection = (pred * target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()
    
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice_loss(pred, target)
        return self.weight_bce * bce_loss + self.weight_dice * dice_loss


# =============================================================================
# Training Pipeline
# =============================================================================

def train_window_detector():
    """Main training function"""
    
    # Hyperparameters
    batch_size = 8
    num_epochs = 50
    learning_rate = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("Training Learnable Window Detector")
    print("="*70)
    print(f"Device: {device}")
    
    # Load dataset
    full_dataset = GapFlytSequenceDataset(
        sequences_dir="../Blender/Outputs/Sequences",
        num_frames=5,
        random_subset=True  # Random sampling during training
    )
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Training sequences: {train_size}")
    print(f"Validation sequences: {val_size}")
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # Models
    flow_extractor = OpticalFlowExtractor(model_type='raft', device=device)
    window_detector = TrainableWindowDetector().to(device)
    
    # Loss and optimizer
    criterion = DiceBCELoss(weight_bce=0.4, weight_dice=0.6)
    optimizer = Adam(window_detector.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    # Training history
    history = {
        'train_loss': [],
        'train_iou': [],
        'val_loss': [],
        'val_iou': []
    }
    
    best_val_iou = 0.0
    
    print("\nStarting training...")
    print("="*70)
    
    for epoch in range(num_epochs):
        # =====================================================================
        # Training Phase
        # =====================================================================
        window_detector.train()
        train_loss = 0.0
        train_iou = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch in pbar:
            frames = batch['frames'].to(device)       # (B, 5, 3, H, W)
            gt_masks = batch['masks'][:, 0].to(device)  # (B, 1, H, W)
            
            # Extract optical flow (frozen pretrained)
            with torch.no_grad():
                F_stack = extract_sequence_flows(flow_extractor, frames)  # (B, 4, 2, H, W)
            
            # Forward pass
            pred_masks = window_detector(F_stack)  # (B, 1, H, W)
            
            # Compute loss
            loss = criterion(pred_masks, gt_masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Compute IoU
            with torch.no_grad():
                intersection = (pred_masks * gt_masks).sum(dim=(1, 2, 3))
                union = ((pred_masks + gt_masks) > 0).float().sum(dim=(1, 2, 3))
                iou = (intersection / (union + 1e-6)).mean()
            
            train_loss += loss.item()
            train_iou += iou.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'iou': f'{iou.item():.4f}'})
        
        train_loss /= len(train_loader)
        train_iou /= len(train_loader)
        
        # =====================================================================
        # Validation Phase
        # =====================================================================
        window_detector.eval()
        val_loss = 0.0
        val_iou = 0.0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]  ')
            for batch in pbar:
                frames = batch['frames'].to(device)
                gt_masks = batch['masks'][:, 0].to(device)
                
                # Extract flow
                F_stack = extract_sequence_flows(flow_extractor, frames)
                
                # Predict
                pred_masks = window_detector(F_stack)
                
                # Loss and IoU
                loss = criterion(pred_masks, gt_masks)
                
                intersection = (pred_masks * gt_masks).sum(dim=(1, 2, 3))
                union = ((pred_masks + gt_masks) > 0).float().sum(dim=(1, 2, 3))
                iou = (intersection / (union + 1e-6)).mean()
                
                val_loss += loss.item()
                val_iou += iou.item()
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'iou': f'{iou.item():.4f}'})
        
        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_iou)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val IoU:   {val_iou:.4f}")
        print("="*70)
        
        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': window_detector.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
            }, 'best_window_detector.pth')
            print(f"✓ Saved new best model (Val IoU: {val_iou:.4f})")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': window_detector.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
            }, f'window_detector_epoch_{epoch+1}.pth')
    
    # Plot training curves
    plot_training_history(history)
    
    print("\n" + "="*70)
    print("Training Complete!")
    print(f"Best Validation IoU: {best_val_iou:.4f}")
    print("="*70)


def plot_training_history(history):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # IoU curves
    axes[1].plot(history['train_iou'], label='Train IoU', linewidth=2)
    axes[1].plot(history['val_iou'], label='Val IoU', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('IoU', fontsize=12)
    axes[1].set_title('Training and Validation IoU', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved training curves to training_history.png")


if __name__ == "__main__":
    train_window_detector()