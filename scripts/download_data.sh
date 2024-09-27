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

kaggle competitions download -c infected-tomato-leaf-vein-segmentation

unzip infected-tomato-leaf-vein-segmentation.zip -d data

echo "Downloading finished, unzipped to 'data'"