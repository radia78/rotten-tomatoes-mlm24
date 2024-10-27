import torch
import os
import pandas as pd
import argparse
from utils.tools import encode_mask, load_and_decode, display_image_and_mask
from utils.data import TomatoLeafDataset
from torch.utils.data import DataLoader
from models import load_model
from matplotlib import pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Predict the tomato leaf mask")
    parser.add_argument('-i', '--image', action='store_true', help="Generate and save prediction images")
    parser.add_argument('model', default="unet", type=str)
    
    args = parser.parse_args()

    TESTDIR = "data/leaf_veins/"
    model_ckpt_dir = f"model_checkpoint/{args.model}"
    model_ckpt_file = os.path.join(model_ckpt_dir, os.listdir(model_ckpt_dir)[-1])
    print(f"Using model checkpoint: {model_ckpt_file}")

    # Load the test dataset and model
    # test_loader = DataLoader(TomatoLeafDataset(TESTDIR + "test.csv", TESTDIR + "test"), batch_size=1)
    test_loader = DataLoader(TomatoLeafDataset(root=TESTDIR, split="test"), batch_size=1)
    # model = TomatoLeafModel()
    model, model_config = load_model(args.model)
    weights = torch.load(model_ckpt_file, weights_only=True)
    model.load_state_dict(weights)
    test_df = pd.read_csv(os.path.join(TESTDIR, "test.csv"))

    # Create predictions for each of image and append it to the csv file
    for sample in test_loader:
        img = sample['image']
        id = sample['id'][0]
        print(f"Predicting for image: {id}")

        pred_mask = model(img)
        test_df.loc[test_df['id'] == id, 'annotation'] = encode_mask(pred_mask.detach(), 0.9)
        # test_df['annotation'] = encode_mask(pred_mask.detach(), 0.9)

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
