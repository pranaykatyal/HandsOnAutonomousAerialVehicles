import PIL
import os
import torch
import torchvision.transforms as T
import random

num_images = 10
augment_steps = 9    # 9 augmentations per image + 1 original
images_folder = "./images/images/"
segmented_folder = "./images/segmented/"

# torch vision works on tensors, but we do not


def guass_noise(input_img):
    inputs = T.ToTensor()(input_img)
    noise = inputs + torch.rand_like(inputs) * random.uniform(0, 1.0)
    noise = torch.clip(noise, 0, 1.)
    output_image = T.ToPILImage()
    image = output_image(noise)
    return image


def blur(input_img):
    blur_transfrom = T.GaussianBlur(
        kernel_size=random.choice([3, 5, 7, 9, 11]), sigma=(0.1, 3.0))
    return blur_transfrom(input_img)


def color_jit(input_img):
    color_jitter = T.ColorJitter(
        brightness=(0.25, 1.75), contrast=(0.25, 2.5),
        saturation=(0.4, 2.0), hue=(-0.2, 0.2))
    return color_jitter(input_img)


# input is an image, and output is ten augmented images
# input size is 640×360
# output sizes will be 256×256 squished
counter = 0
for file in os.listdir(images_folder):
    if file.endswith(".png"):
        print("Working on image: ", file, "out of ", num_images)
        img = PIL.Image.open(images_folder+file)
        img_w, img_h = img.size

        # we just squish the original image
        img_orig = img.resize([256, 256])
        img_orig.save(images_folder+file)

        for aug_idx in range(augment_steps):
            # add gaussian noise
            img_aug = guass_noise(img)
            img_aug = blur(img_aug)
            img_aug = color_jit(img_aug)
            img_aug = img_aug.resize([256, 256])
            img_aug.save(images_folder+file+"_"+str(aug_idx)+".png")

        counter += 1
    if counter >= num_images:
        break

counter = 0
# just squish the masks
for file in os.listdir(segmented_folder):
    if file.endswith(".png"):
        img = PIL.Image.open(segmented_folder+file)
        img_w, img_h = img.size

        img_orig = img.resize([256, 256])
        img_orig.save(segmented_folder+file)

        counter += 1
    if counter >= num_images:
        break
