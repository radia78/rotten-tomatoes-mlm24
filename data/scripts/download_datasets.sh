#!/bin/bash

if [[ ! -f ~/.kaggle/kaggle.json ]]; then
  echo "kaggle.json file does not exist"
  echo -n "Kaggle username: "
  read USERNAME
  echo
  echo -n "Kaggle API key: "
  read APIKEY

  mkdir -p ~/.kaggle
  echo "{\"username\":\"$USERNAME\",\"key\":\"$APIKEY\"}" > ~/.kaggle/kaggle.json
  chmod 600 ~/.kaggle/kaggle.json
fi

echo "Downloading Kaggle competition data..."
kaggle competitions download -p ../ -c infected-tomato-leaf-vein-segmentation
unzip ../infected-tomato-leaf-vein-segmentation.zip -d ../leaf_veins
echo "Downloading finished, unzipped to 'data'"

echo "Downloading retinal vessel competition"
wget -P https://staffnet.kingston.ac.uk/~ku15565/CHASE_DB1/assets/CHASEDB1.zip ../
unzip ../CHASEDB1.zip -d ../retinal_vessel
mkdir ../masks1
ls ../retinal_vessel | grep *HO.png | mv *1stHO.png masks1 && mv *2ndHO.png masks2
