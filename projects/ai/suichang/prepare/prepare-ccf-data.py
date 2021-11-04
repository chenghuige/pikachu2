import os
import os.path as osp
import numpy as np
import cv2
import shutil
from PIL import Image
#import paddlex as pdx
from tqdm.auto import tqdm

# 定义训练集切分时的滑动窗口大小和步长，格式为(W, H)
train_tile_size = (256, 256)
train_stride = (64, 64)
# 定义验证集切分时的滑动窗口大小和步长，格式(W, H)
val_tile_size = (256, 256)
val_stride = (64, 64)

# 下载并解压2015 CCF大数据比赛提供的高清遥感影像
#ccf_remote_dataset = 'https://bj.bcebos.com/paddlex/examples/remote_sensing/datasets/ccf_remote_dataset.tar.gz'
#pdx.utils.download_and_decompress(ccf_remote_dataset, path='./')

if not osp.exists('../input/ccf_remote_dataset/train/image'):
    os.makedirs('../input/ccf_remote_dataset/train/image')
if not osp.exists('../input/ccf_remote_dataset/train/label'):
    os.makedirs('../input/ccf_remote_dataset/train/label')

# 将前4张图片划分入训练集，并切分成小块之后加入到训练集中
# 并生成train_list.txt
for train_id in range(1, 5):
    image = cv2.imread('../input/ccf_remote_dataset/{}.png'.format(train_id))
    label = Image.open('../input/ccf_remote_dataset/{}_class.png'.format(train_id))
    H, W, C = image.shape
    train_tile_id = 1
    hws = []
    for h in range(0, H, train_stride[1]):
        for w in range(0, W, train_stride[0]):
            hws.append((h, w))
    t = tqdm(hws, desc='sub_image')
    for h, w in t:
            left = w
            upper = h
            right = min(w + train_tile_size[0], W)
            lower = min(h + train_tile_size[1], H)
            tile_image = image[upper:lower, left:right, :]
            if tile_image.shape[:2] != train_tile_size:
              continue
            t.set_postfix({'h': tile_image.shape[0], 'w': tile_image.shape[1]})
            #print(train_id, train_tile_id, h, w, tile_image.shape)
            cv2.imwrite("../input/ccf_remote_dataset/train/image/{}_{}.png".format(
                train_id, train_tile_id), tile_image)
            cut_label = label.crop((left, upper, right, lower))
            cut_label.save("../input/ccf_remote_dataset/train/label/{}_{}.png".format(
                train_id, train_tile_id))
            train_tile_id += 1

# 将第5张图片切分成小块之后加入到验证集中
val_id = 5
val_tile_id = 1
image = cv2.imread('../input/ccf_remote_dataset/{}.png'.format(val_id))
label = Image.open('../input/ccf_remote_dataset/{}_class.png'.format(val_id))
H, W, C = image.shape
hws = []
for h in range(0, H, val_stride[1]):
    for w in range(0, W, val_stride[0]):
        hws.append((h, w))
t = tqdm(hws, desc='sub_image')
for h, w in t:
        left = w
        upper = h
        right = min(w + val_tile_size[0], W)
        lower = min(h + val_tile_size[1], H)
        cut_image = image[upper:lower, left:right, :]
        if cut_image.shape[:2] != val_tile_size:
          continue
        #print(val_id, val_title_id, h, w, cut_image.shape)
        t.set_postfix({'h': cut_image.shape[0], 'w': cut_image.shape[1]})
        cv2.imwrite("../input/ccf_remote_dataset/train/image/{}_{}.png".format(
            val_id, val_tile_id), cut_image)
        cut_label = label.crop((left, upper, right, lower))
        cut_label.save("../input/ccf_remote_dataset/train/label/{}_{}.png".format(
            val_id, val_tile_id))
        val_tile_id += 1

# 生成labels.txt
# 植被（标记1）、道路（标记2）、建筑（标记3）、水体（标记4）以及其他(标记0)
label_list = ['background', 'vegetation', 'road', 'build', 'water']

