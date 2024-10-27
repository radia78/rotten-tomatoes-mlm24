import pandas as pd
import os
import torch
import math
import numpy as np
import cv2
import models
from PIL import Image
from torch.utils.data import Dataset
from albumentations import (
    Resize,
    Compose,
    Normalize,
    CoarseDropout,
    Affine,
    GaussNoise,
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
        Normalize()
    ])

forward_transform_mask = Compose([
        Resize(
            transform_image_size(IMAGE_HEIGHT, SCALE), 
            transform_image_size(IMAGE_WIDTH, SCALE),
            interpolation=cv2.INTER_NEAREST_EXACT
        )
    ])

corruption_transforms = Compose([
    GaussNoise(p=0.7, var_limit=(1, 5)),
    CoarseDropout(p=0.7, num_holes_range=(5000, 10000))
])

augmentation_transforms = Compose([
    Affine(rotate=(-360, 360), translate_percent=0.25, p=0.7)
])

transforms_dict = {
    "augmentation_transforms": augmentation_transforms,
    "corruption_transforms": corruption_transforms
}

def rl_decode(enc, shape=(IMAGE_HEIGHT, IMAGE_WIDTH)):
    parts = [int(s) for s in enc.split(' ')]
    dec = list()
    for i in range(0, len(parts), 2):
        cnt = parts[i]
        val = parts[i+1]
        dec += cnt * [val]
    return np.array(dec, dtype=np.uint8).reshape(shape)

class TomatoLeafDataset(Dataset):
    def __init__(self, root: str, split: str="train", transforms=None):
        # Create the directories and open the csv-files
        self.csv_file = f"{root}/{split}.csv"
        self.encodings = pd.read_csv(self.csv_file)
        self.image_dir = f"{root}/{split}"
        self.transforms = transforms
        self.to_tensor = ToTensorV2()

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        idx = idx.tolist() if torch.is_tensor(idx) else idx

        # The find the image name
        img_name = os.path.join(
            self.image_dir,
            self.encodings.iloc[idx, 0] + ".jpg"
        )

        # Try to decode the mask if it exists else set it to 'None'
        try:
            mask = rl_decode(self.encodings.iloc[idx, 1])
        except:
            mask = None
        
        # Convert the image into an array
        img = np.array(Image.open(img_name))

        # Apply preprocess transformation
        img = forward_transform_image(image=img)["image"]
        mask = forward_transform_mask(image=mask)["image"] if mask is not None else None

        # Apply the transformation
        if self.transforms is not None:
            augmented = self.transforms["augmentation_transforms"](image=img, mask=mask)
            img = self.transforms["corruption_transforms"](image=augmented["image"])["image"]

            sample = {"image": self.to_tensor(image=img)["image"], "mask": self.to_tensor(image=augmented["mask"])["image"]}

        else:
            sample = {"image": self.to_tensor(image=img)["image"], "mask": self.to_tensor(image=mask)["image"] if mask is not None else None}

        try:
            assert sample['image'].shape[1] % 32 == 0 and sample['image'].shape[2] % 32 == 0, "Image size must be divisible by 32"
            sample['id'] = self.encodings.iloc[idx, 0]
            return sample
        
        except AssertionError as msg:
            print(msg)
