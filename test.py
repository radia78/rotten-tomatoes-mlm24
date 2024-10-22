import torch
import os
import inspect
import segmentation_models_pytorch as smp
from models.configs import *
from models.dexinet import DexinedSegmenter
from utils.tools import display_image_and_mask
from utils.data import TomatoLeafDataset
import kornia as ki

models_list = os.listdir("model_checkpoint")
model_ckpt_file = models_list[-1]

# Load the dataset
data = TomatoLeafDataset(root="data/leaf_veins")
img = data[5]["image"]

# Load the model
model_config = DexnedSegmenterConfig()
model = DexinedSegmenter(
    classes=model_config.classes,
    activation=model_config.activation,
    pretrained=model_config.pretrained
)
weights = torch.load("model_checkpoint/" + model_ckpt_file, weights_only=True)
model.load_state_dict(weights)
model.to("cpu")
model.eval()
pred_mask = model(img.unsqueeze(0))

# Test the output
# Create the image side by side
display_image_and_mask(img.squeeze(0).permute(1, 2, 0).detach().numpy(), (pred_mask.sigmoid()).squeeze(0).permute(1, 2, 0).detach().numpy(), "model_pred_output_test")
