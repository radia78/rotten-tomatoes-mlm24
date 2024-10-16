from model import *
from ..utils.data_loading import *
from ..utils.utils import *

csv_path = os.path.join("data", "train.csv")
img_dir = os.path.join("data", "train")
dataset = TomatoLeafDataset(csv_file=csv_path, image_dir=img_dir, transform=forward_transform_image)

model = TomatoLeafModel(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    out_classes=1
)

sample_input = dataset[0]['image']
mask_logit = (model(sample_input.unsqueeze(0)).sigmoid() > 0.7).float()

def test_model_output1():
    assert mask_logit.shape == torch.Size([1, 1, 1408, 896])

display_image_and_mask(sample_input.squeeze(0).permute(1, 2, 0).detach().numpy(), mask_logit.squeeze(0).permute(1, 2, 0).detach().numpy(), "model_output_test")
