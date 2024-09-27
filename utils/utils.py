import matplotlib.pyplot as plt
import torch
import os

def display_image_and_mask(image, mask, imgname, figsize=(10, 6)):
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Leaf Sample')

    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.title('Mask')

    if not os.path.exists("test_img_results"):
        os.mkdir("test_img_results")

    plt.savefig(f"test_img_results/{imgname}.jpg")