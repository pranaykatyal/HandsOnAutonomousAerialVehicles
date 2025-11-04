import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import random
from PIL import Image
import os
# from dataaug import *
from Params import *
from utils import *
import pdb

images_folder = "images/"
segmented_folder = "segmented/"
background_folder = "backgrounds/"
# img_size = 256

class WindowDataset(Dataset):
    def __init__(self, ds_path, img_w=256, img_h=256):
        # load all the images and segmented into one large list
        self.DATA = []
        img_path = ds_path + images_folder
        seg_path = ds_path + segmented_folder
        self.background_path = ds_path + background_folder
        self.img_w = img_w
        self.img_h = img_h

        print(self._get_random_background_path())
        for file in os.listdir(img_path):
            if file.endswith(".png"):
                img = Image.open(img_path + file)
                seg = Image.open(seg_path + file)
                img = img.resize([img_w, img_h])
                seg = seg.resize([img_w, img_h])
                self.DATA.append([img, seg])
        print("Dataset initialized")
    def __len__(self):
        N = len(self.DATA * 10)
        return N

    # we do a little trick: to make the 100 000 images, if the image requested number is 47589, we will return image 4758 and a random augmentation (1-9, 0 will return the original)
    def __getitem__(self, idx):
        # idx is from 0 to N-1                
        # Open the RGB image and ground truth label
        # convert them to tensors
        # apply any transform (blur, noise...)

        image_idx = idx // 10
        rgb, label = self.DATA[image_idx]
        # add the random noises if the image is not the original
        if idx % 10 == 0:
            pass
        elif idx % 10 == 1:
            rgb, label = self._get_blank_image()
            rgb = self.add_background(rgb)
            rgb = self.guass_noise(rgb)
            rgb = self.blur(rgb)
            rgb = self.color_jit(rgb)
        else:
            rgb = self.add_background(rgb)
            rgb = self.guass_noise(rgb)
            rgb = self.blur(rgb)
            rgb = self.color_jit(rgb)
        
        # get rid of alpha in the png
        rgb = rgb.convert("RGB")
        rgb = T.ToTensor()(rgb)
        label = label.convert("L")
        label = T.ToTensor()(label)     

        return rgb, label
    
    # !!!TODO NORMLAIZE DATA
    
    def guass_noise(self, input_img):
        inputs = T.ToTensor()(input_img)
        noise = inputs + torch.rand_like(inputs) * random.uniform(0, 1.0)
        noise = torch.clip(noise, 0, 1.)
        output_image = T.ToPILImage()
        image = output_image(noise)
        return image


    def blur(self, input_img):
        blur_transfrom = T.GaussianBlur(
            kernel_size=random.choice([3, 5, 7, 9, 11]), sigma=(0.1, 1.5))
        return blur_transfrom(input_img)


    def color_jit(self, input_img):
        color_jitter = T.ColorJitter(
            brightness=(0.5, 2.0), contrast=(0.33, 3.0),
            saturation=(0.5, 2.0), hue=(-0.35, 0.35))
        return color_jitter(input_img)

    def add_background(self, img):
        # randomly translate and rotate the background image
        background = Image.open(self._get_random_background_path())
        background = background.convert('RGBA')
        background = background.rotate(random.randint(-180, 180))
        background = background.resize([self.img_w, self.img_h])
        # pasting the current image on the selected background
        background.paste(img, mask=img.convert('RGBA'))

        # now do it again, to fill in potential empty corners
        background = Image.open(self._get_random_background_path())
        background = background.convert('RGBA')
        background = background.resize([self.img_w, self.img_h])
        background.paste(img, mask=img.convert('RGBA'))
        return background

    def _get_random_background_path(self):
        # select a random background image
        background_path = random.choice([
            os.path.join(self.background_path, p)
            for p in os.listdir(self.background_path)
            if p.endswith(('jpg'))
        ])
        return background_path
    
    def _get_blank_image(self):
        image = Image.open(self._get_random_background_path())
        image = self.add_background(image)
        mask = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
        mask = Image.fromarray(mask, mode="L")
        return image, mask
    

# verify the dataloader
if __name__ == "__main__":
    dataset = WindowDataset(ds_path=DS_PATH, img_w=256, img_h=256)
    dataloader = DataLoader(dataset)
    import cv2

    rgb, label = dataset[0]
    cv2.imwrite("test0.png" ,CombineImages(rgb,rgb,label))
    rgb, label = dataset[10]
    cv2.imwrite("test10.png" ,CombineImages(rgb,rgb,label))
    rgb, label = dataset[11]
    cv2.imwrite("test11.png" ,CombineImages(rgb,rgb,label))
    rgb, label = dataset[12]
    cv2.imwrite("test12.png" ,CombineImages(rgb,rgb,label))
    rgb, label = dataset[13]
    cv2.imwrite("test13.png" ,CombineImages(rgb,rgb,label))
