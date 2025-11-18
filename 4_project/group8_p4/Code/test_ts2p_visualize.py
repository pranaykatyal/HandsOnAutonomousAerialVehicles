"""
test_ts2p_visualize.py
Visualize TS²P detection on specific sequences
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from ts2p_network import OpticalFlowExtractor, TS2P_GapDetector
from gapflytdataloader import GapFlytSequenceDataset, collate_fn
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize TS²P
flow_extractor = OpticalFlowExtractor(model_type='raft', device=device)
ts2p_detector = TS2P_GapDetector(flow_extractor, device=device)

# Load dataset
dataset = GapFlytSequenceDataset(
    sequences_dir="../Blender/Outputs/Sequences",
    num_frames=5,
    random_subset=False
)

print("="*70)
print("TS²P Interactive Visualization")
print("="*70)
print(f"Total sequences: {len(dataset)}")
print("\nControls:")
print("  - Enter sequence number (0-191) to visualize")
print("  - Enter 'random' for random sequence")
print("  - Enter 'best' to find best IoU sequence")
print("  - Enter 'worst' to find worst IoU sequence")
print("  - Enter 'quit' to exit")
print("="*70)


def visualize_sequence(seq_idx):
    """Visualize detection for a specific sequence"""
    
    # Load sequence
    sample = dataset[seq_idx]
    frames = sample['frames'].unsqueeze(0).to(device)  # (1, 5, 3, H, W)
    masks = sample['masks'].unsqueeze(0).to(device)    # (1, 5, 1, H, W)
    gt_mask = masks[0, 0]  # Reference frame GT
    metadata = sample['metadata']
    seq_name = sample['sequence_name']
    
    print(f"\nProcessing sequence {seq_idx}: {seq_name}")
    print(f"  Texture: {metadata['texture']}")
    print(f"  Mask: {metadata['mask']}")
    print(f"  Rotation: {metadata['rotation_deg']}°")
    print(f"  Scale: {metadata['scale']}")
    
    # Detect gap
    pred_mask, Xi = ts2p_detector.detect_gap(frames, refine=True)
    pred_mask = pred_mask[0]  # (1, H, W)
    Xi = Xi[0]  # (1, H, W)
    
    # Compute IoU
    intersection = (pred_mask * gt_mask).sum()
    union = ((pred_mask + gt_mask) > 0).float().sum()
    iou = (intersection / (union + 1e-6)).item()
    
    print(f"  IoU: {iou:.4f}")
    
    # Create visualization
    fig = plt.figure(figsize=(20, 10))
    
    # Row 1: All 5 frames
    for i in range(5):
        ax = plt.subplot(3, 5, i+1)
        frame = frames[0, i].cpu().permute(1, 2, 0).numpy()
        ax.imshow(frame)
        ax.set_title(f'Frame {i}' + (' (Ref)' if i == 0 else ''), fontsize=10)
        ax.axis('off')
    
    # Row 2: Flow magnitude, prediction, GT, overlay, error
    # Flow magnitude heatmap
    ax = plt.subplot(3, 5, 6)
    im = ax.imshow(Xi[0].cpu().numpy(), cmap='jet')
    ax.set_title('Flow Magnitude Ξ', fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Predicted mask
    ax = plt.subplot(3, 5, 7)
    ax.imshow(pred_mask[0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'Predicted χ', fontsize=10)
    ax.axis('off')
    
    # Ground truth
    ax = plt.subplot(3, 5, 8)
    ax.imshow(gt_mask[0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    ax.set_title('Ground Truth', fontsize=10)
    ax.axis('off')
    
    # Overlay on reference frame
    ax = plt.subplot(3, 5, 9)
    ref_frame = frames[0, 0].cpu().permute(1, 2, 0).numpy()
    ax.imshow(ref_frame)
    # Green = correct, Red = false positive, Blue = false negative
    pred_np = pred_mask[0].cpu().numpy()
    gt_np = gt_mask[0].cpu().numpy()
    overlay = np.zeros((*pred_np.shape, 3))
    overlay[..., 1] = (pred_np > 0.5) & (gt_np > 0.5)  # Green - correct
    overlay[..., 0] = (pred_np > 0.5) & (gt_np < 0.5)  # Red - false positive
    overlay[..., 2] = (pred_np < 0.5) & (gt_np > 0.5)  # Blue - false negative
    ax.imshow(overlay, alpha=0.5)
    ax.set_title(f'Overlay (IoU={iou:.3f})', fontsize=10)
    ax.axis('off')
    
    # Error map
    ax = plt.subplot(3, 5, 10)
    error = torch.abs(pred_mask - gt_mask)[0].cpu().numpy()
    ax.imshow(error, cmap='hot', vmin=0, vmax=1)
    ax.set_title('Error Map', fontsize=10)
    ax.axis('off')
    
    # Row 3: Additional analysis
    # Flow histogram
    ax = plt.subplot(3, 5, 11)
    flow_vals = Xi[0].cpu().numpy().flatten()
    ax.hist(flow_vals, bins=50, alpha=0.7, edgecolor='black')
    threshold_val = np.percentile(flow_vals, ts2p_detector.threshold_percentile)
    ax.axvline(threshold_val, color='r', linestyle='--', linewidth=2, label=f'Threshold (83%)')
    ax.set_xlabel('Flow Magnitude', fontsize=9)
    ax.set_ylabel('Pixel Count', fontsize=9)
    ax.set_title('Flow Distribution', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Stats text
    ax = plt.subplot(3, 5, 12)
    ax.axis('off')
    stats_text = f"""
Sequence: {seq_name}
Texture: {metadata['texture']}
Window: {metadata['mask']}
Rotation: {metadata['rotation_deg']}°
Scale: {metadata['scale']}

IoU: {iou:.4f}
Precision: {(intersection / pred_mask.sum()).item():.4f}
Recall: {(intersection / gt_mask.sum()).item():.4f}

Flow Stats:
  Mean: {flow_vals.mean():.2f}
  Std: {flow_vals.std():.2f}
  Threshold: {threshold_val:.2f}
    """
    ax.text(0.1, 0.5, stats_text, fontsize=9, family='monospace',
            verticalalignment='center')
    
    plt.suptitle(f'TS²P Detection - Sequence {seq_idx}: {seq_name}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    save_path = f'ts2p_visualization_seq_{seq_idx:03d}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.show()
    plt.close()
    
    return iou


def find_best_worst_sequences(num_samples=50):
    """Find sequences with best and worst IoU"""
    print("\nSearching for best/worst sequences (sampling 50 sequences)...")
    
    import random
    sample_indices = random.sample(range(len(dataset)), num_samples)
    
    results = []
    for idx in sample_indices:
        sample = dataset[idx]
        frames = sample['frames'].unsqueeze(0).to(device)
        masks = sample['masks'].unsqueeze(0).to(device)
        gt_mask = masks[0, 0]
        
        pred_mask, _ = ts2p_detector.detect_gap(frames, refine=True)
        pred_mask = pred_mask[0]
        
        intersection = (pred_mask * gt_mask).sum()
        union = ((pred_mask + gt_mask) > 0).float().sum()
        iou = (intersection / (union + 1e-6)).item()
        
        results.append((idx, iou))
        print(f"  Sequence {idx}: IoU = {iou:.4f}")
    
    results.sort(key=lambda x: x[1])
    
    worst_idx, worst_iou = results[0]
    best_idx, best_iou = results[-1]
    
    print(f"\nWorst: Sequence {worst_idx} (IoU = {worst_iou:.4f})")
    print(f"Best: Sequence {best_idx} (IoU = {best_iou:.4f})")
    
    return best_idx, worst_idx


# Interactive loop
while True:
    try:
        user_input = input("\nEnter command: ").strip().lower()
        
        if user_input == 'quit':
            print("Exiting...")
            break
        
        elif user_input == 'random':
            import random
            seq_idx = random.randint(0, len(dataset) - 1)
            visualize_sequence(seq_idx)
        
        elif user_input == 'best':
            best_idx, _ = find_best_worst_sequences()
            visualize_sequence(best_idx)
        
        elif user_input == 'worst':
            _, worst_idx = find_best_worst_sequences()
            visualize_sequence(worst_idx)
        
        else:
            seq_idx = int(user_input)
            if 0 <= seq_idx < len(dataset):
                visualize_sequence(seq_idx)
            else:
                print(f"Invalid sequence number. Must be 0-{len(dataset)-1}")
    
    except ValueError:
        print("Invalid input. Enter a number, 'random', 'best', 'worst', or 'quit'")
    except KeyboardInterrupt:
        print("\nExiting...")
        break