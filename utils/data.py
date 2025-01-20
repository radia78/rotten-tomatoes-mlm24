import pandas as pd
import os
import torch
import math
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from albumentations import (
    Compose,
    Resize,
    Affine,
    ColorJitter
)
from albumentations.pytorch.transforms import ToTensorV2
import lightning as L

'''CONSTANTS'''
IMAGE_HEIGHT = 1400
IMAGE_WIDTH = 875
SCALE = 32

def rl_decode(enc, shape=(IMAGE_HEIGHT, IMAGE_WIDTH)):
    parts = [int(s) for s in enc.split(' ')]
    dec = list()
    for i in range(0, len(parts), 2):
        cnt = parts[i]
        val = parts[i+1]
        dec += cnt * [val]
    return np.array(dec, dtype=np.uint8).reshape(shape)

class BaseSegmentationDataset(Dataset):
    def __init__(self, img_height: int, img_width: int, scale: int=1, transforms=None):
        # Preprocess tranforms
        self.postprocess_img = Compose([
            Resize(
                self.transform_image_size(img_height, 32) // scale, 
                self.transform_image_size(img_width, 32) // scale,
                interpolation=cv2.INTER_CUBIC
            ),
            ToTensorV2()
        ])

        self.postprocess_mask = Compose([
            Resize(
                self.transform_image_size(img_height, 32), 
                self.transform_image_size(img_width, 32),
                interpolation=cv2.INTER_NEAREST_EXACT
            ),
            ToTensorV2()
        ])

        # Additional transforms
        self.transforms = transforms

    def transform_image_size(self, size, scale):
        return math.ceil(size/scale) * scale
    
    def postprocess(self, image, mask):
        return self.postprocess_img(image=image)['image'], self.postprocess_mask(image=mask)['image'] if mask is not None else None

class TomatoLeafDataset(BaseSegmentationDataset):
    def __init__(self, root: str, split: str="train", img_height: int=1400, img_width: int=875, scale: int=1, transforms=None):
        super().__init__(img_height, img_width, scale, transforms)
        # Create the directories and open the csv-files
        self.csv_file = f"{root}/{split}.csv"
        self.encodings = pd.read_csv(self.csv_file)
        self.image_dir = f"{root}/{split}"
        self.split = split

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        idx = idx.tolist() if torch.is_tensor(idx) else idx
        img_id = self.encodings.iloc[idx, 0]

        # The find the image name
        img_name = os.path.join(self.image_dir, img_id + ".jpg")

        # Convert the image to a numpy array
        img = cv2.imread(img_name, cv2.IMREAD_COLOR)

        # If it is a test split
        if self.split == "train":
            mask = rl_decode(self.encodings.iloc[idx, 1])

            # Apply the transformation
            if self.transforms is not None:
                # Geometric transformations
                augmented = self.transforms(image=img, mask=mask)
                final_img, final_mask = self.postprocess(image=augmented['image'], mask=augmented['mask'])

            # Return the ID, image, and mask if it is training
            sample = {
                "id": img_id,
                "image": final_img.float(), 
                "mask": final_mask.long()
            }

        else:
            final_img, _ = self.postprocess(image=img, mask=None)
            # Return the ID and image only if it is testing
            sample = {
                "id": img_id,
                "image": final_img.float() 
            }

        try:
            assert sample['image'].shape[1] % 32 == 0 and sample['image'].shape[2] % 32 == 0, "Image size must be divisible by 32"
            sample['id'] = self.encodings.iloc[idx, 0]
            return sample
        
        except AssertionError as msg:
            print(msg)
        
class BaseSegmentationDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int, scale=1, sdl: bool=False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.scale = scale
        self.num_workers = num_workers

        self.transforms = Compose([
            Affine(
                scale=(1.0, 1.5),
                keep_ratio=True,
                translate_percent=(0.0, 0.5),
                rotate=(-360, 360),
                shear=(-45, 45),
                p=0.7
            ),
            ColorJitter(p=0.7)
        ])

class TomatoLeafDataModule(BaseSegmentationDataModule):
    def setup(self, stage: str=None):
        full_data = TomatoLeafDataset(
            root=self.data_dir,
            split="train",
            scale=self.scale,
            transforms=self.transforms
        )

        self.tomato_train, self.tomato_val = random_split(full_data, [0.8, 0.2])

        self.tomato_predict = TomatoLeafDataset(
            root=self.data_dir,
            split="test",
            scale=self.scale,
            transforms=None
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.tomato_train,
            batch_size=self.batch_size,
            shuffle=True,
            persistent_workers=True,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.tomato_val,
            batch_size=1,
            shuffle=False,
            persistent_workers=True,
            num_workers=self.num_workers
        )
    
    def predict_dataloader(self):
        return DataLoader(
            dataset=self.tomato_predict, 
            batch_size=1,
            shuffle=False,
            persistent_workers=True,
            num_workers=self.num_workers
        )