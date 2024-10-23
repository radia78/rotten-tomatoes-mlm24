import torch
import os
import pandas as pd
from utils.tools import encode_mask
from utils.data import TomatoLeafDataset
from torch.utils.data import DataLoader
from segmentation_models_pytorch import Unet
from models import configs

DIR = "data/"
model_ckpt_file = os.listdir("model_checkpoint")[-1]

TESTDIR = "data/"
model_ckpt_file = os.listdir("model_checkpoint")[-1]
print(f"Using model checkpoint: {model_ckpt_file}")


# Create predictions for each of image and append it to the csv file
for sample in test_loader:
    img = sample['image']
    id = sample['id'][0]
    print(f"Predicting for image: {id}")

    pred_mask = model(img)
    test_df.loc[test_df['id'] == id, 'annotation'] = encode_mask(pred_mask.detach(), 0.9)
    # test_df['annotation'] = encode_mask(pred_mask.detach(), 0.9)

test_df.to_csv("predictions/sample_submission.csv")