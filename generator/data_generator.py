import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import os
from utils.utils import rl_decode, run_length_encode
import argparse


def image_show(path, transform=None, show=True):
    img = Image.open(path)
    if transform:
        img = transform(img)
    if show:
        plt.imshow(img)
        plt.show()
    return img

def mask_transform(source_path, dest_path, transform=None):
    mask_encoded = pd.read_csv(source_path)
    
    for _, row in mask_encoded.iterrows():
        mask = rl_decode(row['annotation'])
        mask = Image.fromarray(mask)
        if transform:
            mask = transform(mask)
        mask_encoded['annotation'] = run_length_encode(mask)
    
    mask_encoded.to_csv(dest_path)

def main():
    train_path = "../data/train/original/"
    mask_path = "../data/test/"
    image_path = os.path.join(train_path, "leaf01.jpg")
    image_show(image_path)


if __name__ == "__main__":
    main()