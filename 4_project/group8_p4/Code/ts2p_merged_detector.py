# ts2p_merged_detector.py

import torch
import numpy as np
import cv2

class TS2PDetectorMerged:
    """
    Merged TS²P-style window/gap detector for multi-frame optical flow sequences.
    Uses your RAFT-based flow_extractor.
    """

    def __init__(self, flow_extractor, device='cuda', detection_resolution=512,
                 num_frames=5, pctile_threshold=15.0,
                 min_area_frac=0.003, max_area_frac=0.5,
                 edge_margin_frac=0.05):
        """
        Args:
            flow_extractor: instance with .compute_flow(frameA, frameB)
            device: torch device
            detection_resolution: downsample size (H=W)
            num_frames: number of frames in the sequence (including reference)
            pctile_threshold: percentile for thresholding low-flow region
            min_area_frac: minimum blob area fraction relative to image
            max_area_frac: maximum blob area fraction
            edge_margin_frac: fraction of image width/height to treat as margin for rejecting near-edge blobs
        """
        self.flow_extractor = flow_extractor
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.detection_resolution = detection_resolution
        self.num_frames = num_frames
        self.pctile_threshold = pctile_threshold
        self.min_area_frac = min_area_frac
        self.max_area_frac = max_area_frac
        self.edge_margin_frac = edge_margin_frac

    

    
    
    
    def detect(self, frames):
        """
        Detect gap from a sequence of frames.
        Args:
            frames: list of (H, W, 3) RGB numpy arrays, length == num_frames
        Returns:
            mask: (H_original, W_original) float32 binary mask (0.0 or 1.0)
            center: (x, y) in original image coordinates or None
            confidence: float in [0,1]
            debug_info: dict of intermediate maps for visualization
        """
        # 1. original size
        H0, W0 = frames[0].shape[:2]

        # 2. downsample frames
        frames_ds = []
        for f in frames:
            f_small = cv2.resize(f, (self.detection_resolution, self.detection_resolution),
                                 interpolation=cv2.INTER_LINEAR)
            frames_ds.append(f_small)

        # 3. convert to tensor and move to device
        # stack flow pairs
        mags = []
        for i in range(len(frames_ds) - 1):
            A = frames_ds[i]
            B = frames_ds[i+1]
            tA = torch.from_numpy(A).float().permute(2,0,1).unsqueeze(0).to(self.device) / 255.0
            tB = torch.from_numpy(B).float().permute(2,0,1).unsqueeze(0).to(self.device) / 255.0
            with torch.no_grad():
                flow = self.flow_extractor.compute_flow(tA, tB)  # shape (1,2,H,W)
                u = flow[0,0]
                v = flow[0,1]
                mag = torch.sqrt(u * u + v * v)
            mags.append(mag.cpu().numpy())

        # 4. aggregate magnitudes
        mags_stack = np.stack(mags, axis=0)  # shape (N-1, H, W)
        mag_agg = np.median(mags_stack, axis=0)  # H×W

        # 5. threshold
        thresh_val = np.percentile(mag_agg.flatten(), self.pctile_threshold)
        # optionally cap threshold (paper uses up to certain max)
        # For safety, we can enforce a cap:
        max_thresh_cap = 60.0
        if thresh_val > max_thresh_cap:
            thresh_val = max_thresh_cap

        binary_init = (mag_agg < thresh_val).astype(np.uint8) * 255

        # 6. morphology: opening and closing
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask_open = cv2.morphologyEx(binary_init, cv2.MORPH_OPEN, kernel_open)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
        mask_clean = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel_close)

        # 7. connected components
        Hds = self.detection_resolution
        Wds = self.detection_resolution
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_clean, connectivity=8)

        # 8. filter & score blobs
        blobs = []
        area_image = Hds * Wds
        min_area = self.min_area_frac * area_image
        max_area = self.max_area_frac * area_image
        edge_margin = int(self.edge_margin_frac * Hds)

        for lbl in range(1, num_labels):
            x, y, w, h, area = stats[lbl]
            if area < min_area or area > max_area:
                continue
            # reject near image edges
            if x < edge_margin or y < edge_margin or x + w > Wds-edge_margin or y + h > Hds-edge_margin:
                continue
            # aspect ratio check
            aspect = float(w) / float(h + 1e-6)
            if aspect < 0.2 or aspect > 5.0:
                continue
            # compute score: area dominates, plus centrality
            cx, cy = centroids[lbl]
            dist_center = np.sqrt((cx - Wds/2.0)**2 + (cy - Hds/2.0)**2)
            max_dist = np.sqrt((Wds/2.0)**2 + (Hds/2.0)**2)
            centrality = 1.0 - (dist_center / max_dist)
            score = area * (0.7 + 0.3 * centrality)
            blobs.append({'label':lbl, 'area':area, 'bbox':(x,y,w,h), 'score':score})

        if not blobs:
            # no valid region
            mask_final_ds = np.zeros((Hds, Wds), dtype=np.float32)
            center = None
            confidence = 0.0
        else:
            # choose best
            blobs = sorted(blobs, key=lambda b: b['score'], reverse=True)
            best = blobs[0]
            lbl = best['label']
            mask_final_ds = (labels == lbl).astype(np.float32)
            x,y,w,h = best['bbox']
            # compute center in downsampled space
            cx = int(x + w/2)
            cy = int(y + h/2)
            # map center to original resolution
            center = (int(cx * (W0/float(Wds))), int(cy * (H0/float(Hds)))
                      )
            # compute confidence = area fraction
            confidence = min(1.0, best['area'] / float(area_image * self.max_area_frac))

        # 9. Upsample mask to original resolution
        mask_upsampled = cv2.resize(mask_final_ds, (W0, H0), interpolation=cv2.INTER_NEAREST)
        mask_upsampled = (mask_upsampled > 0.5).astype(np.float32)

        # 10. debug info
        debug_info = {
            'mag_agg': mag_agg,
            'thresh_val': thresh_val,
            'binary_init': binary_init,
            'mask_clean_ds': mask_clean,
            'labels_ds': labels,
            'frames_ds': frames_ds
        }

        return mask_upsampled, center, confidence, debug_info

def visualize_ts2p_debug(self, debug, out_path):
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    rgb = debug['rgb']
    flow_mag = debug['flow_mag']
    pct_map = debug['pct_map']
    mask_raw = debug['mask_raw']
    mask_clean = debug['mask_clean']
    best_blob = debug['best_blob']
    center = debug['center']

    # Create 2x3 debug grid
    plt.figure(figsize=(16, 10))

    # Panel 1
    plt.subplot(2, 3, 1)
    plt.title("Raw RGB")
    plt.imshow(rgb)
    plt.axis('off')

    # Panel 2
    plt.subplot(2, 3, 2)
    plt.title("Flow Magnitude")
    plt.imshow(flow_mag, cmap='viridis')
    plt.axis('off')

    # Panel 3
    plt.subplot(2, 3, 3)
    plt.title("Percentile Map")
    plt.imshow(pct_map, cmap='plasma')
    plt.axis('off')

    # Panel 4
    plt.subplot(2, 3, 4)
    plt.title("Raw Threshold Mask")
    plt.imshow(mask_raw, cmap='gray')
    plt.axis('off')

    # Panel 5
    plt.subplot(2, 3, 5)
    plt.title("Cleaned Largest Blob")
    plt.imshow(mask_clean, cmap='gray')
    # Draw best blob contour if available
    if best_blob is not None:
        cnt = np.array(best_blob).reshape((-1,1,2)).astype(np.int32)
        overlay = cv2.cvtColor(mask_clean*255, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(overlay, [cnt], -1, (0,255,0), 2)
        plt.imshow(overlay)
    plt.axis('off')

    # Panel 6
    plt.subplot(2, 3, 6)
    plt.title("Final Detection")
    final_overlay = rgb.copy()
    if mask_clean is not None:
        m = (mask_clean > 0)
        final_overlay[m] = (0, 255, 0)
    plt.imshow(final_overlay)

    # Draw center if available
    if center is not None:
        cx, cy = int(center[0]), int(center[1])
        plt.scatter([cx], [cy], c='red', s=50)

    plt.axis('off')

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()