#!/bin/bash

echo "Downloading all the data"
python3 download_data_s3.py

echo "Unzipping Leaf-Veins dataset"
unzip data/infected-tomato-leaf-vein-segmentation.zip -d data/leaf_veins/
echo "Leaf-Veins dataset unzipped at leaf_veins"

echo "Unzipping Retinal-Veins dataset"
unzip data/CHASEBD1.zip -d data/retinal_veins/
echo "Organizing retinal_veins data"
mkdir -p data/retinal_veins/masks1
mkdir -p data/retinal_veins/masks2
mkdir -p data/retinal_veins/img
mv data/retinal_veins/*.jpg data/retinal_veins/img/
mv data/retinal_vesin/*1stHO.png data/retinal_veins/masks1
mv data/retinal_vesin/*2ndHO.png data/retinal_veins/masks2
echo "Finished organizing data"