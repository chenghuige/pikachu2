#!/bin/bash
python -m visdom.server

traindir="dataset/stage1&stage2-train-crf2"
#traindir="dataset/stage2-train"
model_name=pspnet-densenet-s1s2-crf2
CUDA_VISIBLE_DEVICES=0,1,2 python train.py --arch  ${model_name} \
											 --img_rows 256 \
											 --img_cols 256 \
											 --n_epoch 150 \
											 --l_rate 1e-3 \
											 --batch_size 32 \
											 --gpu 0 1 2 \
											 --step 50 \
											 --traindir ${traindir}
										#	 --snapshot snapshot/${model_name}/0.pkl \
										#	 --split "trainval"