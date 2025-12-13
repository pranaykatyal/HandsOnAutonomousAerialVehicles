#----------------------------------------------------------
# now programatically add background images from a folder
images_folder = "./images/images/"
segmented_folder = "./images/segmented/"
import random
import os
from PIL import Image
folder = "./environment/"

for file in os.listdir(images_folder):
    if file.endswith(".png"):
        img = Image.open(images_folder+file)
        img_w, img_h = img.size

        # select a random background image
        background_path = random.choice([
            os.path.join(folder, p)
            for p in os.listdir(folder)
            if p.lower().endswith(('jpg', 'jpeg', 'png'))
        ])

        # randomly translate and rotate the background image
        background = Image.open(background_path)
        background = background.convert('RGBA')
        background = background.rotate(random.randint(-180, 180))
        background = background.resize([img_w, img_h])
        # pasting the current image on the selected background
        background.paste(img, mask=img.convert('RGBA'))
        background.save(images_folder+file)



        # now do it again, to fill in potential empty corners
        img = Image.open(images_folder+file)
        img_w, img_h = img.size

        # select a random background image
        background_path = random.choice([
            os.path.join(folder, p)
            for p in os.listdir(folder)
            if p.lower().endswith(('jpg', 'jpeg', 'png'))
        ])

        # randomly translate and rotate the background image
        background = Image.open(background_path)
        background = background.convert('RGBA')
        background = background.resize([img_w, img_h])
        # pasting the current image on the selected background
        background.paste(img, mask=img.convert('RGBA'))
        background.save(images_folder+file)

    # print a progress update
    print(f"Processed {file}")