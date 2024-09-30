import torch
import os
from segmentation_models_pytorch.metrics import get_stats, iou_score
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
test_loader = TomatoLeafDataset(DIR + "test.csv", DIR + "test")
train_loader = TomatoLeafDataset(DIR + "train.csv", DIR + "train")
image = test_loader[0]['image'].unsqueeze(0)
image = image

# Test the output
pred_mask = (model(image).sigmoid() > 0.9)

# Create the image side by side
display_image_and_mask(image.squeeze(0).permute(1, 2, 0).detach().numpy(), pred_mask.squeeze(0).permute(1, 2, 0).detach().numpy(), "model_pred_output_test")

del image, pred_mask

# Print the Jaccard Index Score for a given mask
metrics_dict = {
    "tp": [],
    "fp": [],
    "fn": [],
    "tn": []
}
for sample in train_loader:
    img, mask = sample['image'].unsqueeze(0), sample['mask'].unsqueeze(0)
    pred_mask = (model(img).sigmoid() > 0.9)

    tp, fp, fn, tn = get_stats(pred_mask, mask, mode="binary")
    
    metrics_dict['tp'].append(tp)
    metrics_dict['fp'].append(fp)
    metrics_dict['fn'].append(fn)
    metrics_dict['tn'].append(tn)

tp_tensor = torch.cat([x for x in metrics_dict['tp']])
fp_tensor = torch.cat([x for x in metrics_dict['fp']])
fn_tensor = torch.cat([x for x in metrics_dict['fn']])
tn_tensor = torch.cat([x for x in metrics_dict['tn']])

jaccard_score_1 = iou_score(tp_tensor, fp_tensor, fn_tensor, tn_tensor, reduction='micro-imagewise')
jaccard_score_2 = iou_score(tp_tensor, fp_tensor, fn_tensor, tn_tensor, reduction='micro')

print("Jaccard Indexes: ")
print(f"Average of each Image: {jaccard_score_1}")
print(f"Average of across all images: {jaccard_score_2}")
