import pandas as pd
import os
import torch
import math
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms.v2 import (
    Compose,
    Resize,
    Normalize,
    RandomAffine,
    GaussianNoise,
    RandomErasing,
    RandomApply,
    InterpolationMode,
    ToTensor
)
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

class TomatoLeafDataset(Dataset):
    def __init__(self, root: str, split: str="train", transforms=None):
        # Create the directories and open the csv-files
        self.csv_file = f"{root}/{split}.csv"
        self.encodings = pd.read_csv(self.csv_file)
        self.image_dir = f"{root}/{split}"
        self.split = split

        # Preprocess tranforms
        self.preprocess_img = Compose([
            ToTensor(),
            Resize(
                (self.transform_image_size(IMAGE_HEIGHT, SCALE), 
                self.transform_image_size(IMAGE_WIDTH, SCALE)),
                interpolation=InterpolationMode.BICUBIC
            ),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        self.preprocess_mask = Compose([
            ToTensor(),
            Resize(
                (self.transform_image_size(IMAGE_HEIGHT, SCALE), 
                self.transform_image_size(IMAGE_WIDTH, SCALE)),
                interpolation=InterpolationMode.NEAREST_EXACT
            )
        ])

        # Additional transforms
        self.transforms = transforms

    def transform_image_size(self, size, scale):
        return math.ceil(size/scale) * scale

    def __len__(self):
        return len(self.encodings)
    
    def preprocess(self, image, mask):
        return self.preprocess_img(image), self.preprocess_mask(mask) if mask is not None else None

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
                aug_img, aug_mask = self.transforms["geometric"](img), self.transforms["geometric"](mask)
                
                # Image based transformations
                img = self.transforms["image"](aug_img)

            # Return the ID, image, and mask if it is training
            sample = {
                "id": img_id,
                "image": aug_img, 
                "mask": aug_mask
            }

        else:
            # Return the ID and image only if it is testing
            img, _ = self.preprocess(img, None)
            sample = {
                "id": img_id,
                "image": img
            }

        try:
            assert sample['image'].shape[1] % 32 == 0 and sample['image'].shape[2] % 32 == 0, "Image size must be divisible by 32"
            sample['id'] = self.encodings.iloc[idx, 0]
            return sample
        
        except AssertionError as msg:
            print(msg)

class RetinalVesselDataset(Dataset):
    def __init__ (self, root: str, transforms: any=None):
        self.transforms = transforms
        self.img_dir = os.path.join(root, "img")
        self.mask_dir = os.path.join(root, "masks1")
        self.resize_mask = Compose([
            ToTensor(),
            Resize((960, 960), interpolation=InterpolationMode.NEAREST_EXACT)
        ])
        self.preprocess_img = Compose([
            ToTensor(),
            Resize((960, 960), interpolation=InterpolationMode.BICUBIC),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(os.listdir(self.img_dir))
    
    def preprocess(self, img, mask):
        return self.preprocess_img(img), self.resize_mask(mask)

    def __getitem__(self, idx):
        idx = idx.tolist() if torch.is_tensor(idx) else idx
        img_name = os.listdir(self.img_dir)[idx]
        mask_name = img_name[:-4] + "_1stHO.png"

        img = cv2.imread(os.path.join(self.img_dir, img_name), cv2.IMREAD_COLOR)
        mask = cv2.imread(os.path.join(self.mask_dir, mask_name), cv2.IMREAD_GRAYSCALE)
        img, mask = self.preprocess(img, mask)

        if self.transforms is not None:
            aug_img, aug_mask = self.transforms['geometric'](img), self.transforms['geometric'](mask) 
            img = self.transforms['image'](aug_img)

            return {'image': aug_img, 'mask': aug_mask}
        
        else:
            return {'image':  img, 'mask': mask}
        
class RetinalVesselDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        image_transforms = Compose([
            RandomApply([GaussianNoise(sigma=0.1),]),
            RandomErasing(p=0.7)
        ])

        geometric_transforms = Compose([
            RandomAffine(degrees=(0, 360), translate=(0.5, 0.5), interpolation=InterpolationMode.BILINEAR)
        ])

        self.transforms = {
            "geometric": geometric_transforms,
            "image": image_transforms
        }

    def setup(self, stage: str=None):
        self.full_dataset = RetinalVesselDataset(self.data_dir, self.transforms)
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

class TomatoLeafDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        image_transforms = Compose([
            RandomApply([GaussianNoise(sigma=0.1),]),
            RandomErasing(p=0.7)
        ])

        geometric_transforms = Compose([
            RandomAffine(degrees=(0, 360), translate=(0.5, 0.5), interpolation=InterpolationMode.BILINEAR)
        ])

        self.transforms = {
            "geometric": geometric_transforms,
            "image": image_transforms
        }

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