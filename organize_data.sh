#!/bin/bash

echo "Downloading all the data"
python3 download_data_s3.py

echo "Unzipping Leaf-Veins dataset"
unzip data/infected-tomato-leaf-vein-segmentation.zip -d data/leaf_veins/
echo "Leaf-Veins dataset unzipped at leaf_veins"
echo "Finished organizing data"