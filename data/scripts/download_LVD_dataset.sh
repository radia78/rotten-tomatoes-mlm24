#! /bin/bash

echo "Downloading the LVD2021 dataset"
mkdir -p ../LVD2021/partitions
for i in {1..5}
do
    echo "Downloading LVD2021 partition $i"
    wget -P https://cloud.tsinghua.edu.cn/d/2dae7d97d90b4259a5df/files/?p=%2FLVD2021.z0$i ../LVD2021/partitions
    unzip ../LVD2021/partitions/LVD2021.z0$i 
done
echo "Downloading the mask of the images"
wget -P https://cloud.tsinghua.edu.cn/d/2dae7d97d90b4259a5df/files/?p=%2FLVD2021.zip ../LVD2021
unzip ../LVD2021/LVD2021.zip