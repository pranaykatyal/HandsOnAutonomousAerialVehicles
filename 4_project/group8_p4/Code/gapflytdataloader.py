"""
PyTorch DataLoader for GapFlyt Parallax Sequences
Example usage for training TS2P network
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import json
import numpy as np


class GapFlytSequenceDataset(Dataset):
    """
    Dataset for loading TS2P diagonal scanning sequences with random subset sampling
    
    Training strategy:
    - Each sequence has 10 frames (1 ref + 9 scan positions)
    - During training: randomly sample 5 frames (1 ref + 4 scan)
    - This teaches the model TS2P logic, not just memorizing one pattern
    
    Each sample contains:
    - frames: tensor of shape (5, C, H, W) - sampled subset
    - masks: tensor of shape (5, 1, H, W) - corresponding GT masks for each frame
    - metadata: dict with sequence info
    """
    
    def __init__(self, sequences_dir, num_frames=5, random_subset=True, transform=None):
        """
        Args:
            sequences_dir: Path to sequences directory
            num_frames: Number of frames to sample (default 5 for deployment)
            random_subset: If True, randomly sample frames during training
            transform: Optional transform
        """
        self.sequences_dir = Path(sequences_dir)
        self.sequences = sorted([d for d in self.sequences_dir.iterdir() if d.is_dir()])
        self.num_frames = num_frames
        self.random_subset = random_subset
        self.transform = transform
        
        if self.transform is None:
            self.transform = T.Compose([
                T.Resize((480, 640)),
                T.ToTensor(),
            ])
        
        print(f"Found {len(self.sequences)} sequences in {sequences_dir}")
        print(f"Sampling strategy: {'Random' if random_subset else 'Fixed'} {num_frames}-frame subsets")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence_dir = self.sequences[idx]
        
        # Load metadata
        with open(sequence_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        total_frames = metadata['num_frames']
        
        # Determine which frames to load
        if self.random_subset and total_frames > self.num_frames:
            # Random sampling: Always include reference frame (0), then sample rest
            import random
            available_scan_frames = list(range(1, total_frames))
            selected_scan_frames = random.sample(available_scan_frames, self.num_frames - 1)
            frame_indices = [0] + sorted(selected_scan_frames)  # Keep reference first
        else:
            # Fixed sampling: Use first num_frames
            frame_indices = list(range(min(self.num_frames, total_frames)))
        
        # Load selected frames and their corresponding masks
        frames = []
        masks = []
        for i in frame_indices:
            # Load RGB frame
            frame_path = sequence_dir / f"frame_{i:02d}.png"
            img = Image.open(frame_path).convert('RGB')
            img_tensor = self.transform(img)
            frames.append(img_tensor)
            
            # Load corresponding mask
            mask_path = sequence_dir / f"mask_{i:02d}.png"
            mask = Image.open(mask_path).convert('L')
            mask_tensor = self.transform(mask)
            masks.append(mask_tensor)
        
        # Stack frames: (N, C, H, W)
        frames = torch.stack(frames)
        masks = torch.stack(masks)  # (N, 1, H, W)
        
        return {
            'frames': frames,              # (N, 3, H, W) - sampled subset
            'masks': masks,                # (N, 1, H, W) - corresponding masks
            'frame_indices': frame_indices,  # Which frames were selected
            'metadata': metadata,
            'sequence_name': sequence_dir.name
        }


def collate_fn(batch):
    """
    Custom collate function for TS2P sequences
    """
    return {
        'frames': torch.stack([item['frames'] for item in batch]),        # (B, N, C, H, W)
        'masks': torch.stack([item['masks'] for item in batch]),          # (B, N, 1, H, W)
        'frame_indices': [item['frame_indices'] for item in batch],       # List of sampled indices
        'metadata': [item['metadata'] for item in batch],
        'sequence_name': [item['sequence_name'] for item in batch]
    }


# Example usage
if __name__ == "__main__":
    
    # Create dataset with random subset sampling (for training)
    train_dataset = GapFlytSequenceDataset(
        sequences_dir="../Blender/Outputs/Sequences",
        num_frames=5,        # Sample 5 frames (1 ref + 4 scan)
        random_subset=True   # Randomly sample different combinations
    )
    
    # Create dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    print("\n" + "="*70)
    print("Training Mode: Random 5-frame subsets from 10 available")
    print("="*70)
    
    # Show a few batches to demonstrate variety
    for batch_idx, batch in enumerate(train_dataloader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Frames shape: {batch['frames'].shape}")      # (4, 5, 3, 480, 640)
        print(f"  Masks shape: {batch['masks'].shape}")        # (4, 5, 1, 480, 640)
        
        # Show which frames were sampled for each sequence
        for i in range(min(2, len(batch['sequence_name']))):
            frames = batch['frames'][i]
            indices = batch['frame_indices'][i]
            meta = batch['metadata'][i]
            
            print(f"\n    Sequence {i+1}: {batch['sequence_name'][i]}")
            print(f"      Sampled frame indices: {indices}")
            print(f"      (Always includes frame 0 as reference)")
            print(f"      Texture: {meta['texture']} | Rotation: {meta['rotation_deg']}°")
        
        if batch_idx == 2:  # Show 3 batches
            break
    
    print("\n" + "="*70)
    print("Why this strategy?")
    print("  - Each epoch sees DIFFERENT 5-frame combinations")
    print("  - Model learns TS2P principle, not memorizing patterns")
    print("  - Better generalization to unseen scanning trajectories")
    print("  - Each frame has its own GT mask from that camera viewpoint")
    print("="*70)
    
    # Also show fixed sampling (for validation/testing)
    print("\n" + "="*70)
    print("Validation Mode: Fixed 5-frame subset")
    print("="*70)
    
    val_dataset = GapFlytSequenceDataset(
        sequences_dir="../Blender/Outputs/Sequences",
        num_frames=5,
        random_subset=False  # Always use same frames for validation
    )
    
    val_dataloader = DataLoader(val_dataset, batch_size=4, collate_fn=collate_fn)
    
    batch = next(iter(val_dataloader))
    print(f"Frames shape: {batch['frames'].shape}")
    print(f"Masks shape: {batch['masks'].shape}")
    print(f"Fixed indices: Always uses frames [0, 1, 2, 3, 4]")
    
    print("\n" + "="*70)
    print(f"Dataset ready! Total sequences: {len(train_dataset)}")
    print("="*70)