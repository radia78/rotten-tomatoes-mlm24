import torch
import os
import pandas as pd
import argparse
from utils.tools import encode_mask, load_and_decode, display_image_and_mask
from utils.data import TomatoLeafDataset
from utils.training import load_model
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser(description="Predict the tomato leaf mask")
    parser.add_argument('model', type=str, default="unet", help="Specified model name to be tested")
    parser.add_argument('-i', '--image', action='store_true', help="Generate and save prediction images")
    args = parser.parse_args()

    model_ckpt_file = "model_checkpoint/data_augmented_weights2.pt"
    print(f"Using model checkpoint: {model_ckpt_file}")

    dataset = TomatoLeafDataset(
        root="data/leaf_veins",
        split="test", 
        transforms=None
    )

    # Load the test dataset and model
    test_loader = DataLoader(
            dataset=dataset, 
            batch_size=1, 
            shuffle=False,
            pin_memory=False
    )
    model, model_config = load_model(args.model)
    weights = torch.load(model_ckpt_file, weights_only=True)
    model.load_state_dict(weights)
    model.to("cpu")
    model.eval()
    test_df = pd.read_csv("data/leaf_veins/" + "test.csv")

    # Create predictions for each of image and append it to the csv file
    for sample in test_loader:
        img = sample['image']
        id = sample['id'][0]
        print(f"Predicting for image: {id}")

        with torch.no_grad():
            pred_mask = model(img)
        test_df.loc[test_df['id'] == id, 'annotation'] = encode_mask(pred_mask.detach(), 0.9)

    if not os.path.exists("predictions"):
        os.makedirs("predictions")

    test_df.to_csv("predictions/sample_submission.csv")

    if args.image:
        if not os.path.exists("test_img_results/test"):
            os.makedirs("test_img_results/test")

        images, masks = load_and_decode("predictions/sample_submission.csv", "data/leaf_veins/test")
        for i in range(len(images)):
            name = test_df.iloc[i]['id']
            display_image_and_mask(images[i], masks[i], name, "predictions/images")

if __name__ == "__main__":
    main()
