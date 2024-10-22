from utils.tools import load_and_decode, display_image_and_mask

imgs, masks = load_and_decode("predictions/sample_submission.csv", img_dir="data/leaf_veins/test")
display_image_and_mask(image=imgs[4], mask=masks[4], imgname="submission_results")