"""
ts2p_network.py - TS²P (Temporally Stacked Spatial Parallax) Network
Direct learning from multi-view sequences to window segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from gapflytdataloader import GapFlytSequenceDataset, collate_fn


class SpatialParallaxEncoder(nn.Module):
    """
    Extracts features from individual frames
    Uses ResNet-style encoder
    """
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Encoder blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.conv2 = self._make_layer(64, 128, blocks=2)
        self.conv3 = self._make_layer(128, 256, blocks=2)
        self.conv4 = self._make_layer(256, 512, blocks=2)
        
    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        layers.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        for _ in range(blocks - 1):
            layers.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # x: (B, C, H, W)
        x1 = self.conv1(x)   # 1/4 resolution
        x2 = self.conv2(x1)  # 1/8 resolution
        x3 = self.conv3(x2)  # 1/16 resolution
        x4 = self.conv4(x3)  # 1/32 resolution
        return [x1, x2, x3, x4]


class TemporalParallaxFusion(nn.Module):
    """
    Fuses features from multiple views using attention
    Key idea: Window regions have consistent appearance across views,
    background changes due to parallax
    """
    def __init__(self, feature_dim=512):
        super().__init__()
        
        # Cross-view attention
        self.query = nn.Conv2d(feature_dim, feature_dim // 8, kernel_size=1)
        self.key = nn.Conv2d(feature_dim, feature_dim // 8, kernel_size=1)
        self.value = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        
        # Fusion convolution
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, ref_features, scan_features):
        """
        Args:
            ref_features: (B, C, H, W) - Reference frame features
            scan_features: (B, N-1, C, H, W) - Scan frames features
        Returns:
            fused_features: (B, C, H, W)
        """
        B, N_minus_1, C, H, W = scan_features.shape
        
        # Compute attention between reference and each scan frame
        Q = self.query(ref_features)  # (B, C//8, H, W)
        
        # Average attention across all scan frames
        attended_features = []
        for i in range(N_minus_1):
            scan_feat = scan_features[:, i]  # (B, C, H, W)
            K = self.key(scan_feat)
            V = self.value(scan_feat)
            
            # Attention
            Q_flat = Q.view(B, C // 8, -1).permute(0, 2, 1)  # (B, HW, C//8)
            K_flat = K.view(B, C // 8, -1)  # (B, C//8, HW)
            
            attention = torch.softmax(torch.bmm(Q_flat, K_flat) / np.sqrt(C // 8), dim=-1)  # (B, HW, HW)
            
            V_flat = V.view(B, C, -1)  # (B, C, HW)
            attended = torch.bmm(V_flat, attention.permute(0, 2, 1))  # (B, C, HW)
            attended = attended.view(B, C, H, W)
            attended_features.append(attended)
        
        # Average attended features
        avg_attended = torch.stack(attended_features, dim=1).mean(dim=1)  # (B, C, H, W)
        
        # Fuse with reference
        fused = torch.cat([ref_features, avg_attended], dim=1)  # (B, 2C, H, W)
        fused = self.fusion_conv(fused)  # (B, C, H, W)
        
        return fused


class TS2PDecoder(nn.Module):
    """
    Decoder with skip connections for segmentation
    """
    def __init__(self):
        super().__init__()
        
        # Upsampling blocks
        self.up1 = self._make_up_block(512, 256)
        self.up2 = self._make_up_block(256, 128)
        self.up3 = self._make_up_block(128, 64)
        self.up4 = self._make_up_block(64, 32)
        
        # Final prediction
        self.final = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def _make_up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features):
        # features: list of [x1, x2, x3, x4] from encoder
        x = features[-1]  # Start from deepest features
        
        x = self.up1(x)  # 1/16 -> 1/8
        x = self.up2(x)  # 1/8 -> 1/4
        x = self.up3(x)  # 1/4 -> 1/2
        x = self.up4(x)  # 1/2 -> 1/1
        
        mask = self.final(x)
        return mask


class TS2PNetwork(nn.Module):
    """
    Complete TS²P Network
    Input: Multi-view image sequence (B, N, 3, H, W)
    Output: Window segmentation mask (B, 1, H, W)
    """
    def __init__(self):
        super().__init__()
        
        self.encoder = SpatialParallaxEncoder(in_channels=3)
        self.temporal_fusion = TemporalParallaxFusion(feature_dim=512)
        self.decoder = TS2PDecoder()
        
    def forward(self, frames):
        """
        Args:
            frames: (B, N, 3, H, W) - N frames (1 reference + N-1 scan)
        Returns:
            mask: (B, 1, H, W) - Window segmentation
        """
        B, N, C, H, W = frames.shape
        
        # Extract features from all frames
        all_features = []
        for i in range(N):
            frame = frames[:, i]  # (B, 3, H, W)
            features = self.encoder(frame)  # List of multi-scale features
            all_features.append(features[-1])  # Use deepest features
        
        # Reference frame (first) and scan frames (rest)
        ref_features = all_features[0]  # (B, 512, H/32, W/32)
        scan_features = torch.stack(all_features[1:], dim=1)  # (B, N-1, 512, H/32, W/32)
        
        # Temporal fusion using parallax cues
        fused_features = self.temporal_fusion(ref_features, scan_features)
        
        # Decode to segmentation mask
        # For simplicity, pass fused features through decoder
        # In practice, you'd use skip connections from encoder
        mask = self.decoder([None, None, None, fused_features])
        
        # Upsample to input resolution
        mask = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False)
        
        return mask


def train_ts2p():
    """Training script for TS²P network"""
    
    # Hyperparameters
    batch_size = 4
    num_epochs = 50
    learning_rate = 1e-4
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
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # Model
    model = TS2PNetwork().to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Training TS²P Network on {device}")
    print(f"Total sequences: {len(train_dataset)}")
    print(f"Batch size: {batch_size}")
    print("="*70)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            frames = batch['frames'].to(device)  # (B, 5, 3, 480, 640)
            masks = batch['masks'].to(device)    # (B, 5, 1, 480, 640)
            
            # Use reference frame mask as target (frame 0)
            target_mask = masks[:, 0]  # (B, 1, 480, 640)
            
            # Forward pass
            pred_mask = model(frames)  # (B, 1, 480, 640)
            
            # Compute loss
            loss = criterion(pred_mask, target_mask)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'ts2p_checkpoint_epoch_{epoch+1}.pth')
    
    print("Training complete!")
    torch.save(model.state_dict(), 'ts2p_final.pth')


if __name__ == "__main__":
    train_ts2p()