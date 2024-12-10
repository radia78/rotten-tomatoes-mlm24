import matplotlib.pyplot as plt
from torchvision.transforms.v2 import Resize, InterpolationMode
from kornia.morphology import closing
from utils.data import IMAGE_HEIGHT, IMAGE_WIDTH, rl_decode
import torch
import os
import pandas as pd
from PIL import Image
import numpy as np

kernel = torch.ones(3, 3)

def display_image_and_mask(image, mask, imgname, save_dir='test_img_results', figsize=(10, 6)):
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Leaf Sample')

    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.title('Mask')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(f"{save_dir}/{imgname}.jpg")

resize_mask = Resize((IMAGE_HEIGHT, IMAGE_WIDTH), InterpolationMode.NEAREST_EXACT)

def run_length_encode(mask):
    enc = []
    cache_val = mask[0]
    val_counter = 0

    for i in range(len(mask)):
        if cache_val != mask[i]:
            enc.append(str(val_counter))
            enc.append(str(cache_val))
            val_counter = 0

        cache_val = mask[i]
        val_counter += 1

        if i == (len(mask) - 1):
            enc.append(str(val_counter))
            enc.append(str(cache_val))

    return " ".join(enc)

def encode_mask(mask: torch.Tensor, threshold: float):
    # Turn into binary mask first
    mask = (mask.sigmoid() > threshold).long()

    # Close the holes and gaps
    mask = closing(mask, kernel.to(mask.device))

    # Resize the mask
    mask = resize_mask(mask)

    # Turn mask into list
    mask = mask.flatten().tolist()

    return run_length_encode(mask)

def load_and_decode(train_csv, img_dir):
    train_data = pd.read_csv(train_csv)
    images, masks = [], []
    for _, row in train_data.iterrows():
        img_path = f"{img_dir}/{row['id']}.jpg"
        img = Image.open(img_path)
        mask = rl_decode(row['annotation'])
        images.append(np.array(img))
        masks.append(mask)
    return images, masks

def load_images(train_csv, img_dir, to_np_array=True):
    train_data = pd.read_csv(train_csv)
    images = []
    for _, row in train_data.iterrows():
        id = row['id']
        img_path = f"{img_dir}/{id}.jpg"
        img = Image.open(img_path)
        images.append((id, np.array(img) if to_np_array else img))
    return images