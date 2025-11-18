"""
tune_ts2p_threshold.py
Find optimal threshold percentile for TS²P gap detection
"""
from ts2p_network import *
from gapflytdataloader import GapFlytSequenceDataset, collate_fn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset (use subset for speed)
dataset = GapFlytSequenceDataset(
    sequences_dir="../Blender/Outputs/Sequences",
    num_frames=5,
    random_subset=False
)

# Use first 48 sequences for tuning (12 batches)
import itertools
dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
dataloader = list(itertools.islice(dataloader, 12))  # 48 sequences

# Test different thresholds
thresholds = [70, 72, 75, 77, 80, 83, 86, 88, 90]
results = {}

print("Testing different threshold percentiles...")
print("="*70)

for threshold_pct in thresholds:
    # Create new detector for each threshold
    flow_extractor = OpticalFlowExtractor(model_type='raft', device=device)
    ts2p_detector = TS2P_GapDetector(flow_extractor, device=device)
    ts2p_detector.threshold_percentile = threshold_pct
    
    total_iou = 0.0
    num_samples = 0
    
    for batch in dataloader:
        frames = batch['frames'].to(device)
        gt_masks = batch['masks'][:, 0].to(device)
        
        # Detect gap with current threshold
        pred_masks, _ = ts2p_detector.detect_gap(frames, refine=True)
        
        # Compute IoU
        intersection = (pred_masks * gt_masks).sum(dim=[1, 2, 3])
        union = ((pred_masks + gt_masks) > 0).float().sum(dim=[1, 2, 3])
        iou = (intersection / (union + 1e-6)).mean()
        
        total_iou += iou.item()
        num_samples += 1
    
    avg_iou = total_iou / num_samples
    results[threshold_pct] = avg_iou
    print(f"Threshold {threshold_pct:2d}%: IoU = {avg_iou:.4f}")

print("="*70)
best_threshold = max(results, key=results.get)
best_iou = results[best_threshold]
print(f"\nBest threshold: {best_threshold}% → IoU = {best_iou:.4f}")
print(f"Improvement: {best_iou - results[50]:+.4f} over default (50%)")
print(f"Current (50%): {results[50]:.4f}")
print("="*70)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(list(results.keys()), list(results.values()), 'bo-', linewidth=2, markersize=8)
plt.axvline(x=best_threshold, color='r', linestyle='--', linewidth=2, label=f'Best: {best_threshold}%')
plt.axhline(y=best_iou, color='r', linestyle=':', alpha=0.5)
plt.axvline(x=50, color='gray', linestyle='--', linewidth=1, label='Default: 50%')
plt.xlabel('Threshold Percentile', fontsize=12)
plt.ylabel('Average IoU', fontsize=12)
plt.title('TS²P Threshold Tuning Results', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.ylim([min(results.values()) - 0.02, max(results.values()) + 0.02])
plt.tight_layout()
plt.savefig('threshold_tuning_results.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved plot to threshold_tuning_results.png")

# Save results to text file
with open('threshold_tuning_results.txt', 'w') as f:
    f.write("TS²P Threshold Tuning Results\n")
    f.write("="*70 + "\n\n")
    for threshold_pct, iou in sorted(results.items()):
        marker = " ← BEST" if threshold_pct == best_threshold else ""
        f.write(f"Threshold {threshold_pct:2d}%: IoU = {iou:.4f}{marker}\n")
    f.write("\n" + "="*70 + "\n")
    f.write(f"Best threshold: {best_threshold}%\n")
    f.write(f"Best IoU: {best_iou:.4f}\n")
    f.write(f"Improvement: {best_iou - results[50]:+.4f} over default\n")

print("✓ Saved results to threshold_tuning_results.txt")
print("\nNext step: Update ts2p_network.py with best threshold:")
print(f"    self.threshold_percentile = {best_threshold}  # in TS2P_GapDetector.__init__()")