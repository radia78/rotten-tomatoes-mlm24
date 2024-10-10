import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import os
from utils.utils import rl_decode, encode_mask_img, load_images
import argparse


def image_show(path, transform=None, show=True):
    img = Image.open(path)
    if transform:
        img = transform(img)
    if show:
        plt.imshow(img)
        plt.show()
    return img

def mask_transform(csv_path, dest_path, transform=None, output_name="train.csv"):
    mask_encoded = pd.read_csv(csv_path)
    
    for _, row in mask_encoded.iterrows():
        mask = rl_decode(row['annotation'])
        mask = Image.fromarray(mask)
        if transform:
            mask = transform(mask)
        mask_encoded['annotation'] = encode_mask_img(mask)
    
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    dest_path = os.path.join(dest_path, output_name)

    mask_encoded.to_csv(dest_path)

def image_transform(source_path, dest_path, transform=None):
    train_csv_path = os.path.join(source_path, "train.csv")
    images = load_images(train_csv_path, source_path, to_np_array=False)
    for image_id, image in images:
        if transform:
            image = transform(image)
            # print(f"Hi from image {image_id}")
            # plt.imshow(image)
            # plt.show()
        image.save(dest_path + image_id + ".jpg")

def main():
    train_path = "data/train/"
    mask_csv_path = os.path.join(train_path, "original/train.csv")
    image_path = os.path.join(train_path, "original/")

    dest_dir = os.path.join(train_path, "vertical_flipped/")       
    
    transform = transforms.RandomVerticalFlip(p=1) # Why Random?

    mask_transform(mask_csv_path, dest_dir, transform=transform)
    image_transform(image_path, dest_dir, transform=transform)
    


if __name__ == "__main__":
    main()