import torch
from torch import nn
from torchvision import transforms as T
import cv2
import numpy as np

# from network import *
# from utils import *
# TRAINED_MODEL = ""

# model = Network(3, 1)
# model.load_state_dict(torch.load(TRAINED_MODEL, weights_only=True))
# model = model.to(device)
# model.eval()


class Window_Segmentaion():
    def __init__(self, torch_network, model_path, model_thresh ,in_ch, out_ch, img_w, img_h, use_depth=True):
        self.img_w = img_w
        self.img_h = img_h
        self.model_thresh = model_thresh
        datatype = torch.float32
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('device', self.device )
        self.model = torch_network(in_ch, out_ch)
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.to(self.device )
        self.model.eval()


    def get_pred(self, rgb):

                # Resize to model input size
        if isinstance(rgb, np.ndarray):
            rgb_resized = cv2.resize(rgb, (self.img_w, self.img_h))
            
            # Convert BGR to RGB if needed
            if rgb_resized.shape[2] == 3:
                rgb_resized = cv2.cvtColor(rgb_resized, cv2.COLOR_BGR2RGB)
        else: #its PIL
            rgb_resized = rgb.resize((self.img_w, self.img_h))
        cv2.imwrite(f'pre_seg_view.png', rgb_resized)
        rgb_tensor = T.ToTensor()(rgb_resized)
        # Add batch dimension
        rgb_tensor = rgb_tensor.unsqueeze(0)  # Shape: [1, C, H, W]

        rgb_tensor = rgb_tensor.to(self.device)
        print(f"sending rgb image shape {rgb_tensor.shape}")

        # Disable gradients for inference (saves memory and speeds up)
        with torch.no_grad():
            pred_logits = self.model(rgb_tensor)  # Shape: [1, out_ch, H, W]
            # Apply sigmoid to convert logits to probabilities
            pred_probs = torch.sigmoid(pred_logits)
  
        # Remove batch dimension and move to CPU
        pred_probs = pred_probs.squeeze(0).detach().cpu()  # Shape: [out_ch, H, W] or [H, W]
        # Convert to numpy
        if pred_probs.shape[0] == 1:  # If single channel output
            pred_probs = pred_probs.squeeze(0).numpy()  # Shape: [H, W]
        else:
            pred_probs = pred_probs.numpy()  # Shape: [out_ch, H, W]
        
        # Convert probabilities to binary mask (threshold at 0.5) and scale to 0-255
        return ((pred_probs > self.model_thresh).astype(np.uint8) * 255)

    
        #asssuming input images are already opencv
    def get_closest_frame(self, rgb, depth):
        #resize depth to be the same size as model output
        depth=cv2.resize(depth, (self.img_w, self.img_h))
        window_mask = self.get_pred(rgb)
        # Ensure mask is binary 0/1
        window_mask01 = (window_mask > 0).astype(np.uint16)
        print(f"depth shape {depth.shape}, max:{depth.max()}, min, {depth.min()}")
        print(f"window_mask shape {window_mask.shape}, max:{window_mask.max()}, min, {window_mask.min()}")

        #mask our depth image 
        window_depth = depth * window_mask01
        #find centerr of all closed contores

        window_depth_normalized = cv2.normalize(window_depth, None, 0, 255, cv2.NORM_MINMAX)
        window_depth_normalized = window_depth_normalized.astype(np.uint8)
        cv2.imwrite(f'images/window_depth_normalized.png', window_depth_normalized)


        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            window_mask01.astype(np.uint8), connectivity=8, ltype=cv2.CV_32S
        )
        # print(f'num_labels{num_labels}, labels{labels}, stats{stats}, centroids{centroids} ')
        areas = stats[1:, cv2.CC_STAT_AREA]
        order_by_area = np.argsort(areas)[::-1] + 1 
        # largest component's label id
        largest_id = order_by_area[0]

        # mask for largest blob
        largest_mask = (labels == largest_id)

        largest_window_mask = depth * largest_mask
        largest_window_mask_normalized = cv2.normalize(largest_window_mask, None, 0, 255, cv2.NORM_MINMAX)
        largest_window_mask_normalized = largest_window_mask_normalized.astype(np.uint8)
        cv2.imwrite(f'images/largest_window_mask_normalized.png', largest_window_mask_normalized)
        # # bounding box and centroid of largest blob
        # x = stats[largest_id, cv2.CC_STAT_LEFT]
        # y = stats[largest_id, cv2.CC_STAT_TOP]
        # w = stats[largest_id, cv2.CC_STAT_WIDTH]
        # h = stats[largest_id, cv2.CC_STAT_HEIGHT]
        (cw, ch) = centroids[largest_id]
        # print(f'x{x}')
        # print(f'y{y}')
        # print(f'w{w}')
        # print(f'h{h}')
        print(f'cw{cw}, ch{ch}')
        avg_depth = largest_window_mask[largest_window_mask>0].mean()
        print(f'avg depth {avg_depth}')

        ex = avg_depth
        ey = cw - (self.img_w/2)
        ez = ch - (self.img_h/2)
        return ex, ey, ez        #then erode depth image arround each respective depth

    # def save_video(self)

    # def clean_segmentation