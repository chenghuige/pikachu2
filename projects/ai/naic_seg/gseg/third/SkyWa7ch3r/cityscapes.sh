#!/bin/bash

#Make the directories
sudo mkdir -p ~/datasets/data ~/datasets/zips ~/datasets/data/cityscapes

#Goto the zips directory
cd ~/datasets/zips

#Login to the Cityscapes Site
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=<username>&password=<password>&submit=Login' https://www.cityscapes-dataset.com/login/
#Get the Fine Ground Truths 241MB    
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
#Get the 8 bit images associated with those ground truths 11GB
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3
#Get the Ground Truth Coarse Images 1.3GB
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=2
#Get the Images for the Coarse ground Truths 44GB
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=4

#Unzip all the data
unzip gtFine_trainvaltest.zip -d <absolute-dir-to-cityscapes>
unzip leftImg8bit_trainvaltest.zip -d <absolute-dir-to-cityscapes>
unzip gtCoarse.zip -d <absolute-dir-to-cityscapes>
unzip leftImg8bit_trainextra.zip -d <absolute-dir-to-cityscapes>









