import torch
import os
import pandas as pd
from utils.utils import encode_mask
from utils.data_loading import TomatoLeafDataset
from torch.utils.data import DataLoader
from unet.model import TomatoLeafModel

DIR = "data/"
model_ckpt_file = os.listdir("model_checkpoint")[-1]

# Load the test dataset and model
test_loader = DataLoader(TomatoLeafDataset(DIR + "test.csv", DIR + "test"), batch_size=1)
model = TomatoLeafModel()
weights = torch.load("model_checkpoint/" + model_ckpt_file, weights_only=True)
model.load_state_dict(weights)
test_df = pd.read_csv(DIR + "test.csv")

# Create predictions for each of image and append it to the csv file
for sample in test_loader:
    img = sample['image']

    pred_mask = model(img)
    test_df['annotation'] = encode_mask(pred_mask.detach(), 0.9)

if not os.path.exists("predictions"):
    os.makedirs("predictions")

test_df.to_csv("predictions/sample_submission.csv")