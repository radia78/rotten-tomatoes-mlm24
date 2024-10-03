import matplotlib.pyplot as plt
from torchvision.transforms.v2 import Resize, InterpolationMode
from utils.data_loading import IMAGE_HEIGHT, IMAGE_WIDTH
import cv2
import torch
import os
import pandas as pd
from PIL import Image
import numpy as np

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
    # Resize the mask first
    mask = resize_mask(mask)

    # Apply transformations
    mask = (mask.sigmoid() > threshold).long().flatten().tolist()

    return run_length_encode(mask)

def rl_decode(enc, shape=(1400, 875)):
    parts = [int(s) for s in enc.split(' ')]
    dec = list()
    for i in range(0, len(parts), 2):
        cnt = parts[i]
        val = parts[i+1]
        dec += cnt * [val]
    return np.array(dec, dtype=np.uint8).reshape(shape)

def load_and_decode(train_csv, img_dir):
    train_data = pd.read_csv(train_csv)
    images, masks = [], []
    for idx, row in train_data.iterrows():
        img_path = f"{img_dir}/{row['id']}.jpg"
        img = Image.open(img_path)
        mask = rl_decode(row['annotation'])
        images.append(np.array(img))
        masks.append(mask)
    return images, masks

images, masks = load_and_decode('data/train.csv', 'data/train')