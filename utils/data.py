import pandas as pd
import os
import torch
import math
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader, random_split
from albumentations import (
    Compose,
    Normalize,
    Resize,
    CoarseDropout,
    GaussNoise,
    Rotate,
    VerticalFlip,
    HorizontalFlip
)
from albumentations.pytorch.transforms import ToTensorV2
import lightning as L

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
    def __init__(self, img_height: int, img_width: int, transforms=None):
        # Preprocess tranforms
        self.preprocess_img = Compose([
            Resize(
                self.transform_image_size(img_height, 32), 
                self.transform_image_size(img_width, 32),
                interpolation=cv2.INTER_CUBIC
            ),
            Normalize()
        ])

        self.preprocess_mask = Compose([
            Resize(
                self.transform_image_size(img_height, 32), 
                self.transform_image_size(img_width, 32),
                interpolation=cv2.INTER_NEAREST_EXACT
            )
        ])

        # Additional transforms
        self.transforms = transforms
        self.to_tensor = ToTensorV2()

    def transform_image_size(self, size, scale):
        return math.ceil(size/scale) * scale
    
    def preprocess(self, image, mask):
        return self.preprocess_img(image=image)['image'], self.preprocess_mask(image=mask)['image'] if mask is not None else None

class TomatoLeafDataset(BaseSegmentationDataset):
    def __init__(self, root: str, split: str="train", img_height: int=1400, img_width: int=875, transforms=None):
        super().__init__(img_height, img_width, transforms)
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
            # Apply preprocess transformation
            img, mask = self.preprocess(img, mask)

            # Apply the transformation
            if self.transforms is not None:
                # Geometric transformations
                augmented = self.transforms["geometric"](image=img, mask=mask)
                
                # Image based transformations
                img = self.transforms["image"](image=augmented['image'])

            # Return the ID, image, and mask if it is training
            sample = {
                "id": img_id,
                "image": self.to_tensor(image=img['image'])['image'], 
                "mask": self.to_tensor(image=augmented['mask'])['image']
            }

        else:
            # Return the ID and image only if it is testing
            img, _ = self.preprocess(img, None)
            sample = {
                "id": img_id,
                "image": self.to_tensor(image=img)['image']
            }

        try:
            assert sample['image'].shape[1] % 32 == 0 and sample['image'].shape[2] % 32 == 0, "Image size must be divisible by 32"
            sample['id'] = self.encodings.iloc[idx, 0]
            return sample
        
        except AssertionError as msg:
            print(msg)

class RetinalVesselDataset(BaseSegmentationDataset):
    def __init__ (self, root: str, img_height: int=960, img_width: int=999, transforms: any=None):
        super().__init__(img_height, img_width, transforms)
        self.img_dir = os.path.join(root, "img")
        self.mask_dir = os.path.join(root, "masks1")

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        idx = idx.tolist() if torch.is_tensor(idx) else idx
        img_name = os.listdir(self.img_dir)[idx]
        mask_name = img_name[:-4] + "_1stHO.png"

        img = cv2.imread(os.path.join(self.img_dir, img_name), cv2.IMREAD_COLOR)
        mask = cv2.imread(os.path.join(self.mask_dir, mask_name), cv2.IMREAD_GRAYSCALE)
        img, mask = self.preprocess(img, mask)

        if self.transforms is not None:
            augmented = self.transforms['geometric'](image=img, mask=mask)
            img = self.transforms['image'](image=augmented['image'])

            return {'image': self.to_tensor(image=img['image'])['image'], 'mask': self.to_tensor(image=augmented['mask'])['image']}
        
        else:
            return {'image':  self.to_tensor(image=img)['image'], 'mask': self.to_tensor(image=mask)['image']}
        
class BaseSegmentationDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        image_transforms = Compose([
            GaussNoise(var_limit=(10, 50), p=0.7),
            CoarseDropout(
                max_holes=5_000,
                min_holes=1_000,
                max_height=16,
                min_height=4,
                max_width=16,
                min_width=8,
                p=0.7
            )
        ])

        geometric_transforms = Compose([
            Rotate(limit=(-360, 360), p=0.7),
            HorizontalFlip(p=0.7),
            VerticalFlip(p=0.7)
            
        ])
        self.transforms = {
            "geometric": geometric_transforms,
            "image": image_transforms
        }
        
class RetinalVesselDataModule(BaseSegmentationDataModule):
    def setup(self, stage: str=None):
        self.full_dataset = RetinalVesselDataset(root=self.data_dir, transforms=self.transforms)
        self.train_data, self.val_data = random_split(self.full_dataset, [0.8, 0.2])

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True
        )

class TomatoLeafDataModule(BaseSegmentationDataModule):
    def setup(self, stage: str=None):
        self.tomato_train = TomatoLeafDataset(
            root=self.data_dir,
            split="train",
            transforms=self.transforms
        )

        self.tomato_predict = TomatoLeafDataset(
            root=self.data_dir,
            split="test",
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
    
    def predict_dataloader(self):
        return DataLoader(
            dataset=self.tomato_predict, 
            batch_size=1,
            shuffle=False,
            persistent_workers=True,
            num_workers=self.num_workers
        )