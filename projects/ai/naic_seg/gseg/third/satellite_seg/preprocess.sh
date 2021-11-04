#!/bin/bash

ProjectDir=.

echo "convert stage1 trainging dataset to stage2 format..."
python utils/convert_s1_to_s2.py
echo "convert label format done"

echo "preprocess the training dataset with crf..."

#stage2
train_dir2=${ProjectDir}/dataset/CCF-training-Semi
python utils/inference.py ${train_dir2}/1.png ${train_dir2}/1_visual.png ${train_dir2}/1_visual_crf.png 0.95 5 &
python utils/inference.py ${train_dir2}/2.png ${train_dir2}/2_visual.png ${train_dir2}/2_visual_crf.png 0.95 5 &
python utils/inference.py ${train_dir2}/3.png ${train_dir2}/3_visual.png ${train_dir2}/3_visual_crf.png 0.95 5 &

wait
echo "generating dataset..."
python utils/generate_data.py
echo "Done"