import torch
import os
from utils.utils import display_image_and_mask
from utils.data_loading import TomatoLeafDataset
from unet.model import TomatoLeafModel

DIR = "data/"
models_list = os.listdir("model_checkpoint")
model_ckpt_file = models_list[-1]

# Load the model
model = TomatoLeafModel()
weights = torch.load("model_checkpoint/" + model_ckpt_file, weights_only=True)
model.load_state_dict(weights)
model.to("cpu")
model.eval()

# Load the dataset
dataloader = TomatoLeafDataset(DIR + "test.csv", DIR + "test")
image = dataloader[0]['image'].unsqueeze(0)
image = image.to("cpu")

# Test the output
pred_mask = (model(image).sigmoid() > 0.9)

# Create the image side by side
display_image_and_mask(image.squeeze(0).permute(1, 2, 0).detach().numpy(), pred_mask.squeeze(0).permute(1, 2, 0).detach().numpy(), "model_pred_output_test")
