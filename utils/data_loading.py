import pandas as pd
import os
import torch
import math
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
from albumentations import (
    Resize,
    Compose,
    Normalize
)
from albumentations.pytorch import ToTensorV2

IMAGE_HEIGHT = 1400
IMAGE_WIDTH = 875
SCALE = 32

# Image height for processing
def transform_image_size(size: int, scale: int):
    return math.ceil(size/scale)*scale

# Image transform functions
forward_transform_image = Compose([
        Resize(
            transform_image_size(IMAGE_HEIGHT, SCALE), 
            transform_image_size(IMAGE_WIDTH, SCALE),
            interpolation=cv2.INTER_CUBIC
        ),
        Normalize(),
        ToTensorV2()
    ])

forward_transform_mask = Compose([
        Resize(
            transform_image_size(IMAGE_HEIGHT, SCALE), 
            transform_image_size(IMAGE_WIDTH, SCALE),
            interpolation=cv2.INTER_NEAREST_EXACT
        ),
        ToTensorV2()
    ])

def rl_decode(enc, shape=(IMAGE_HEIGHT, IMAGE_WIDTH)):
    parts = [int(s) for s in enc.split(' ')]
    dec = list()
    for i in range(0, len(parts), 2):
        cnt = parts[i]
        val = parts[i+1]
        dec += cnt * [val]
    return np.array(dec, dtype=np.uint8).reshape(shape)

class TomatoLeafDataset(Dataset):
    def __init__(self, csv_file: str, image_dir: str, transform=None):
        self.csv_file = csv_file
        self.encodings = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        idx = idx.tolist() if torch.is_tensor(idx) else idx

        img_name = os.path.join(
            self.image_dir,
            self.encodings.iloc[idx, 0] + ".jpg"
        )

        try:
            mask = rl_decode(self.encodings.iloc[idx, 1])

        except:
            mask = None

        img = np.array(Image.open(img_name))

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        

        # Image dim is H, W, C
        sample = forward_transform_image(image=img)

        try:
            assert sample['image'].shape[1] % 32 == 0 and sample['image'].shape[2] % 32 == 0, "Image size must be divisible by 32"
            sample['mask'] = forward_transform_mask(image=mask)['image'] if mask is not None else [0]

            return sample

        
        except AssertionError as msg:
            print(msg)
            