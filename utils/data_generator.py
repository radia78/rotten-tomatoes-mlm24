import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import os
from utils.utils import rl_decode, encode_mask_img, load_images
from albumentations import (Compose, 
                            HorizontalFlip,
                            GridDropout, 
                            ShiftScaleRotate)

augmentation_transforms = Compose([
    HorizontalFlip(p=0.5),
    GridDropout(p=0.5),
    ShiftScaleRotate(p=0.5, shift_limit=0.0625, scale_limit=0.1, rotate_limit=45)
], additional_targets={"mask": "mask"})

def data_transform(source_path, mask_csv_path, dest_path, transform=None):
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    
    mask_encoded = pd.read_csv(mask_csv_path)
    
    for _, row in mask_encoded.iterrows():
        mask = rl_decode(row['annotation'])
        # mask = Image.fromarray(mask)
        image_id = row['id']
        img_path = f"{source_path}/{image_id}.jpg"
        img = np.array(Image.open(img_path))
        if transform:
            augmented = transform(image=img, mask=mask)
            mask = augmented['mask']
            img = augmented['image']
        Image.fromarray(img).save(dest_path + image_id + ".jpg")
            
        row['annotation'] = encode_mask_img(Image.fromarray(mask))

    dest_path = os.path.join(dest_path, "train.csv")
    mask_encoded.to_csv(dest_path)


def main():
    train_path = "data/train/"
    mask_csv_path = os.path.join(train_path, "original/train.csv")
    image_path = os.path.join(train_path, "original/")

    name = "composed_test"
    dest_dir = os.path.join(train_path, f"{name}/")       
    
    transform = augmentation_transforms

    # mask_transform(mask_csv_path, dest_dir, transform=transform)
    # image_transform(image_path, dest_dir, transform=transform)
    
    data_transform(image_path, mask_csv_path, dest_dir, transform=transform)


if __name__ == "__main__":
    main()