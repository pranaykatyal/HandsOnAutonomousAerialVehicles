import torch
import argparse
import numpy as np
from PIL import Image
import cv2
import os
import glob

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent / 'RAFT' / 'core'))

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder


class RaftFlow():
    def __init__(self,model_pth:str,alternative_corr:bool, small_model:bool=False):
        self.frame_count = 0
        self.clear_old_visualizations()

        model_params = argparse.Namespace(
            small=small_model,
            dropout=False,
            alternative_corr=alternative_corr,
            mixed_precision=False,
            model=model_pth,
        )
        print(f'selected model params: {model_params}')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('device', self.device)
        self.model = torch.nn.DataParallel(RAFT(model_params))
        self.model.load_state_dict(torch.load(model_params.model))
        self.model.eval()


    def get_flow(self, img1, img2, iters=20):
        with torch.no_grad():
            img1 = self._load_image(img1)
            img2 = self._load_image(img2)

            padder = InputPadder(img1.shape)
            img1, img2 = padder.pad(img1, img2)
            # print(f'image sizes: {img1.shape} 2: {img2.shape}')

            flow_low, flow_up = self.model(img1, img2, iters=iters, test_mode=True)
            # self._vixulize(img1, flow_up)
            return flow_up[0].cpu().numpy()

    def _load_image(self, img):
        img = np.array(img).astype(np.uint8)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img[None].to(self.device)
    
    def _vixulize(self, img, flo):
        img = img[0].permute(1,2,0).cpu().numpy()
        flo = flo[0].permute(1,2,0).cpu().numpy()
        # print(f'flow shape: {flo.shape}, min:{flo.min()}, max:{flo.max()}, \n')
        # map flow to rgb image
        flo = flow_viz.flow_to_image(flo)
        img_flo = np.concatenate([img, flo], axis=0)
        # print(f' mromalized{flo} \n flow shape: {flo.shape}, min:{flo.min()}, max:{flo.max()}, \n')
        # import matplotlib.pyplot as plt
        # cv2.imwrite('image2.png',img_flo)
        # plt.show()

        cv2.imwrite('flow_image.png', img_flo[:, :, [0,1,2]])
        # cv2.waitKey()
    
        print('getting flow')

    def get_closest_frame(self, segment):
        

        # Ensure mask is binary 0/1
        segment_01 = (segment > 0).astype(np.uint16)
        # print(f"window_mask shape {segment.shape}, max:{segment.max()}, min, {segment.min()}")
        H, W = segment_01.shape

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            segment_01.astype(np.uint8), connectivity=8, ltype=cv2.CV_32S
        )
        closed_labels = []

        for label in range(1, num_labels):  # skip 0 (background)
            x, y, w, h, area = stats[label]
            y0, y1 = y, y + h - 1
            x0, x1 = x, x + w - 1

            # Reject blobs that touch the image border
            if y0 == 0 or x0 == 0 or y1 == H - 1 or x1 == W - 1:
                continue

            closed_labels.append(label)

        if not closed_labels:
            # No closed zero-blob found
            return np.zeros_like(segment_01, dtype=np.uint8), None

        # Pick the blob with the largest area
        largest_label = max(
            closed_labels,
            key=lambda lbl: stats[lbl, cv2.CC_STAT_AREA]
        )

        largest_mask = (labels == largest_label).astype(np.uint8)
        # where x is the column (horizontal/width) and y is the row (vertical/height)
        cx, cy = centroids[largest_label]
        
        # Calculate errors from image center
        # ey: horizontal error (x-direction)
        # ez: vertical error (y-direction)
        ey = cx - (W / 2)  # cx is column position, W is width
        ez = cy - (H / 2)  # cy is row position, H is height

        return largest_mask, (ey,ez)
    

    def get_displacement_from_img_pair(self,img1,img2):
        flow_image = self.get_flow(img1,img2)
        # print('got flow')
        #squish flow
        # squished_flow = flow_image.sum(axis=0)
        # print(f'squished_flow shape {squished_flow.shape}, squeezed {squished_flow.squee}')
        u = flow_image[0]
        v = flow_image[1]
        mag_flow_image = np.sqrt(u**2 + v**2 + 1e-8)
        # print(f'mag_flow_image shape {mag_flow_image.shape}')

        #save flow image 
        # norm_sumed_flow_image = cv2.normalize(mag_flow_image.astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX)
        # cv2.imwrite('squeezed_flow.png', norm_sumed_flow_image)

        #threshold to get just small flow
        min_flow = mag_flow_image.min()
        max_flow = mag_flow_image.max()
        flow_threshold = min_flow + np.abs(max_flow-min_flow) * .35
        # print(f'flow shape {flow_image.shape}')
        # avg_flow = mag_flow_image.mean()

        #mask based off threshold
        mask = mag_flow_image < flow_threshold

        #save mask
        # print(f'mask shape {mask.shape}')
        # norm_mask = cv2.normalize(mask.astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX)
        # cv2.imwrite('masked_hole_thresh.png', norm_mask)
        # cv2.imwrite('mask2.png', norm_mask[1])

        #get largest closed blob
        largest_mask, (ey,ez) = self.get_closest_frame(mask)

        #save largest mask for debigging
        # norm_largest_mask = cv2.normalize(largest_mask.astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX)
        # cv2.imwrite('norm_largest_mask.png', norm_largest_mask)
        self.save_overlay_image(img1, largest_mask, ey, ez)
        self.save_visulized_flow_frame(img1, flow_image, mask, largest_mask, ey, ez)

        return ey,ez

    def save_visulized_flow_frame(self, input_img, flow, masked_flow, largest_mask, ey, ez):
        """
        Save a composite image showing input, flow, masked flow, and largest mask side by side.
        
        Parameters:
        - input_img: RGB input image (H, W, 3)
        - flow: Raw flow image (2, H, W) with u,v components
        - masked_flow: Binary mask (H, W) showing thresholded flow regions
        - largest_mask: Binary mask (H, W) showing largest closed blob
        - ey: Error in y direction (pixels)
        - ez: Error in z direction (pixels)
        """
        # Convert flow to RGB visualization using flow_viz (same as _vixulize)
        flow_rgb = flow_viz.flow_to_image(flow.transpose(1, 2, 0))  # (H, W, 2) -> RGB
        
        # Normalize masks for visualization
        masked_flow_viz = cv2.normalize(masked_flow.astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX)
        masked_flow_viz = cv2.cvtColor(masked_flow_viz, cv2.COLOR_GRAY2BGR)
        
        largest_mask_viz = cv2.normalize(largest_mask.astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX)
        largest_mask_viz = cv2.cvtColor(largest_mask_viz, cv2.COLOR_GRAY2BGR)
        
        # Ensure input image is BGR for OpenCV
        input_img_bgr = input_img.astype(np.uint8)
        
        # Get target dimensions from input image
        target_h, target_w = input_img_bgr.shape[:2]
        
        # Resize all images to match input dimensions
        flow_rgb = cv2.resize(flow_rgb, (target_w, target_h))
        masked_flow_viz = cv2.resize(masked_flow_viz, (target_w, target_h))
        largest_mask_resized = cv2.resize(largest_mask, (target_w, target_h))
        largest_mask_viz = cv2.resize(largest_mask_viz, (target_w, target_h))
        
        # Calculate the actual centroid of the blob
        moments = cv2.moments(largest_mask_resized)
        if moments['m00'] != 0:
            blob_center_x = int(moments['m10'] / moments['m00'])
            blob_center_y = int(moments['m01'] / moments['m00'])
        else:
            # Fallback if blob has no area
            blob_center_x = target_w // 2
            blob_center_y = target_h // 2
        
        # Draw red dot at actual blob center
        cv2.circle(largest_mask_viz, (blob_center_x, blob_center_y), radius=5, color=(0, 0, 255), thickness=-1)
        
        # Draw crosshair at image center for reference (green)
        cv2.line(largest_mask_viz, (target_w//2 - 10, target_h//2), (target_w//2 + 10, target_h//2), (0, 255, 0), 2)
        cv2.line(largest_mask_viz, (target_w//2, target_h//2 - 10), (target_w//2, target_h//2 + 10), (0, 255, 0), 2)
        
        # Add error text to top-left corner
        error_text = f"ey: {ey:.1f}px, ez: {ez:.1f}px"
        cv2.putText(largest_mask_viz, error_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255, 255, 0), 2, cv2.LINE_AA)
        
        # Create labels
        label_height = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        labels = ["Input Image", "Flow Visualization", "Masked Flow", "Largest Blob"]
        
        # Create label images
        labeled_images = []
        for img, label in zip([input_img_bgr, flow_rgb, masked_flow_viz, largest_mask_viz], labels):
            # Create white label bar
            label_bar = np.ones((label_height, target_w, 3), dtype=np.uint8) * 255
            
            # Add text to label bar
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            text_x = (target_w - text_size[0]) // 2
            text_y = (label_height + text_size[1]) // 2
            cv2.putText(label_bar, label, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)
            
            # Concatenate label and image vertically
            labeled_img = np.vstack([label_bar, img])
            labeled_images.append(labeled_img)
        
        # Concatenate all images horizontally
        composite_image = np.hstack(labeled_images)
        
        # Save with frame counter
        filename = f'run/flow_visualization_{self.frame_count:04d}.png'
        cv2.imwrite(filename, composite_image)
        print(f'Saved visualization: {filename}')
        
        self.frame_count += 1


    def save_overlay_image(self, input_img, largest_mask, ey, ez):
        """
        Save RGB input image with largest mask overlayed in red with alpha blending.
        
        Parameters:
        - input_img: RGB input image (H, W, 3)
        - largest_mask: Binary mask (H, W) showing largest closed blob
        - ey: Error in y direction (pixels)
        - ez: Error in z direction (pixels)
        """
        # Ensure input image is RGB
        if input_img.shape[2] == 3:
            input_img_rgb = input_img.astype(np.uint8).copy()
        else:
            input_img_rgb = cv2.cvtColor(input_img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        
        # Get dimensions
        target_h, target_w = input_img_rgb.shape[:2]
        
        # Resize mask to match input dimensions if needed
        if largest_mask.shape != (target_h, target_w):
            largest_mask_resized = cv2.resize(largest_mask, (target_w, target_h))
        else:
            largest_mask_resized = largest_mask
        
        # Create red overlay
        red_overlay = np.zeros_like(input_img_rgb)
        red_overlay[:, :, 0] = 255  # Red channel
        
        # Apply mask to overlay
        mask_3channel = np.stack([largest_mask_resized] * 3, axis=-1).astype(bool)
        
        # Alpha blend: output = input * (1 - alpha) + overlay * alpha
        alpha = 0.5
        output_img = input_img_rgb.copy()
        output_img[mask_3channel] = (
            input_img_rgb[mask_3channel] * (1 - alpha) + 
            red_overlay[mask_3channel] * alpha
        ).astype(np.uint8)
        
        # Calculate centroid for visualization
        moments = cv2.moments(largest_mask_resized)
        if moments['m00'] != 0:
            blob_center_x = int(moments['m10'] / moments['m00'])
            blob_center_y = int(moments['m01'] / moments['m00'])
        else:
            blob_center_x = target_w // 2
            blob_center_y = target_h // 2
        
        # Draw red dot at blob center
        cv2.circle(output_img, (blob_center_x, blob_center_y), radius=5, color=(255, 0, 0), thickness=-1)
        
        # Draw crosshair at image center (green)
        cv2.line(output_img, (target_w//2 - 10, target_h//2), (target_w//2 + 10, target_h//2), (0, 255, 0), 2)
        cv2.line(output_img, (target_w//2, target_h//2 - 10), (target_w//2, target_h//2 + 10), (0, 255, 0), 2)
        
        # Add error text to top-left corner
        error_text = f"ey: {ey:.1f}px, ez: {ez:.1f}px"
        cv2.putText(output_img, error_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 255, 0), 2, cv2.LINE_AA)
        
        # Convert RGB to BGR for OpenCV saving
        output_img_bgr = output_img# cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        
        # Save with frame counter
        filename = f'run/overlay_{self.frame_count:04d}.png'
        cv2.imwrite(filename, output_img_bgr)
        print(f'Saved overlay: {filename}')

    def clear_old_visualizations(self):
        """
        Delete all old flow visualization images from the run directory.
        """
        # Create run directory if it doesn't exist
        os.makedirs('run', exist_ok=True)
        
        # Find all flow visualization images
        old_files = glob.glob('run/*.png')
        
        # Delete each file
        for file_path in old_files:
            try:
                os.remove(file_path)
                print(f'Deleted old visualization: {file_path}')
            except Exception as e:
                print(f'Error deleting {file_path}: {e}')
        
        if old_files:
            print(f'Cleared {len(old_files)} old visualization files')
        else:
            print('No old visualization files to clear')