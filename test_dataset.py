from utils.data import TomatoLeafDataset, transform_image_size, IMAGE_HEIGHT, IMAGE_WIDTH, SCALE, transforms_dict
from utils.tools import display_image_and_mask
import numpy as np
import torch

dataset = TomatoLeafDataset(root="data/leaf_veins", transforms=transforms_dict)

def test_data_loading_1():
    for sample in dataset:
        assert sample['image'].shape == torch.Size([3, transform_image_size(IMAGE_HEIGHT, SCALE), transform_image_size(IMAGE_WIDTH, SCALE)])

def test_data_loading_2():
    for sample in dataset:
        assert sample['mask'].shape == torch.Size([1, transform_image_size(IMAGE_HEIGHT, SCALE), transform_image_size(IMAGE_WIDTH, SCALE)])

def test_data_loading_3():
    for sample in dataset:
        assert np.sum(sample['mask'].shape) > 0

sample = dataset[0]
display_image_and_mask(sample['image'].permute(1, 2, 0).numpy(), sample['mask'].permute(1, 2, 0).numpy(), "test_sample_leaf")
