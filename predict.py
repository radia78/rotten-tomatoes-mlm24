import torch
import os
import pandas as pd
import argparse
from utils.utils import encode_mask, load_and_decode, display_image_and_mask
from utils.data_loading import TomatoLeafDataset
from torch.utils.data import DataLoader
from unet.model import TomatoLeafModel

def main():
    parser = argparse.ArgumentParser(description="Predict the tomato leaf mask")
    parser.add_argument('-i', '--image', action='store_true', help="Generate and save prediction images")
    args = parser.parse_args()

    TESTDIR = "data/test"
    model_ckpt_file = os.listdir("model_checkpoint")[-1]

    # Load the test dataset and model
    test_loader = DataLoader(TomatoLeafDataset(TESTDIR + "test.csv", TESTDIR + "test"), batch_size=1)
    model = TomatoLeafModel()
    weights = torch.load("model_checkpoint/" + model_ckpt_file, weights_only=True)
    model.load_state_dict(weights)
    test_df = pd.read_csv(TESTDIR + "test.csv")

    # Create predictions for each of image and append it to the csv file
    for sample in test_loader:
        img = sample['image']

        pred_mask = model(img)
        test_df['annotation'] = encode_mask(pred_mask.detach(), 0.9)

    if not os.path.exists("predictions"):
        os.makedirs("predictions")

    test_df.to_csv("predictions/sample_submission.csv")

    if args.image:
        images, masks = load_and_decode("predictions/sample_submission.csv", TESTDIR + "test")
        for i in range(len(images)):
            name = test_df.iloc[i]['id']
            display_image_and_mask(images[i], masks[i], name, "predictions/images")

           

if __name__ == "__main__":
    main()