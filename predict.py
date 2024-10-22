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

# Load the test dataset and model
test_loader = TomatoLeafDataset(root="data/leaf_veins")
model_config = configs.SMPUnetConfig()
model = Unet(
    encoder_name=model_config.encoder_name,
    encoder_weights=model_config.encoder_weights,
    in_channels=model_config.in_channels,
    decoder_use_batchnorm=model_config.decoder_use_batchnorm,
    decoder_attention_type=model_config.decoder_attention_type,
    classes=model_config.classes
)
weights = torch.load("model_checkpoint/" + model_ckpt_file, weights_only=True)
model.load_state_dict(weights)
test_df = pd.read_csv("data/leaf_veins/test.csv")

# Create predictions for each of image and append it to the csv file
for sample in test_loader:
    img = sample['image']

    pred_mask = model(img.unsqueeze(0))
    test_df['annotation'] = encode_mask(pred_mask.detach(), 0.5)

if not os.path.exists("predictions"):
    os.makedirs("predictions")

test_df.to_csv("predictions/sample_submission.csv")