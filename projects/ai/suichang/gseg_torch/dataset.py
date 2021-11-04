#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   dataset.py
#        \author   chenghuige  
#          \date   2020-10-16 16:38:41.301623
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import cv2

from absl import flags
FLAGS = flags.FLAGS

import numpy as np
from torch.utils import data

import albumentations as A

from .third import segmentation_models_pytorch as smp

from gseg.config import *

def get_train_augmentation():
  train_transforms = [
    # A.JpegCompression(p=0.5),
    # A.Rotate(limit=80, p=1.0),
    # A.OneOf([
    #     A.OpticalDistortion(),
    #     A.GridDistortion(),
    #     A.IAAPiecewiseAffine(),
    # ]),
    # A.RandomSizedCrop(min_max_height=(int(resolution*0.7), input_res),
    #                   height=resolution, width=resolution, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    # A.GaussianBlur(p=0.3),
    # A.OneOf([
    #     A.RandomBrightnessContrast(),   
    #     A.HueSaturationValue(),
    # ]),
    # A.Cutout(num_holes=8, max_h_size=resolution//8, max_w_size=resolution//8, fill_value=0, p=0.3),
  ]
  return A.Compose(train_transforms)

# def get_valid_transforms():
#     return A.Compose([
#             A.CenterCrop(height=resolution, width=resolution, p=1.0),
#             A.Normalize(),
#             ToTensorV2(),
#         ], p=1.0)

# def get_tta_transforms():
#     return A.Compose([
#             A.JpegCompression(p=0.5),
#             A.RandomSizedCrop(min_max_height=(int(resolution*0.9), int(resolution*1.1)),
#                               height=resolution, width=resolution, p=1.0),
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.5),
#             A.Transpose(p=0.5),
#             A.Normalize(),
#             ToTensorV2(),
#         ], p=1.0)

def get_augmentation(mode):
  if mode == 'train':
    return get_train_augmentation()
  else:
    return None

def to_tensor(x, **kwargs):
  return x.transpose(2, 0, 1).astype('float32')

def to_mask_tensor(mask, **kwargs):
  return mask.astype('float32')

def get_preprocessing(preprocessing_fn):
  _transform = [
    A.Lambda(image=preprocessing_fn),
    A.Lambda(image=to_tensor, mask=to_mask_tensor),
  ]
  return A.Compose(_transform)

def _get_img_mask_ids(images_path, masks_path):
  res = []
  for dir_entry in os.listdir(images_path):
    if os.path.isfile(os.path.join(images_path, dir_entry)):
      img_id, _ = os.path.splitext(dir_entry)
      res.append((os.path.join(images_path, img_id + ".tif"),
                  os.path.join(masks_path, img_id + ".png"),
                  img_id))
  return res

class Dataset(data.Dataset):
  def __init__(self, folder, mode, augmentation=None, preprocessing=None, seed=12345):
    super(Dataset).__init__()
    np.random.seed(seed)
    l =  _get_img_mask_ids(f'{folder}/image', f'{folder}/label')
    np.random.shuffle(l)
    images, masks, image_ids = zip(*l)
    self.images, self.masks, self.image_ids = [], [], []

    records, folds = 40, 10
    for i in range(len(images)):
      index = i % records # 40 records TODO 注意这里和tfrecord生成耦合 包括seed确保一致
      fold = index % folds
      if FLAGS.fold is None or mode == 'test' \
        or (mode == 'valid' and fold == FLAGS.fold) \
            or (mode == 'train' and fold != FLAGS.fold):
        self.images.append(images[i])
        self.masks.append(masks[i])
        self.image_ids.append(image_ids[i])

    self.augmentation = augmentation or get_augmentation(mode)
    preprocessing_fn = smp.encoders.get_preprocessing_fn(FLAGS.backbone, FLAGS.backbone_weights)
    self.preprocessing = preprocessing or get_preprocessing(preprocessing_fn)

    logging.info(mode, folder, len(self), len(images), len(self.images), FLAGS.fold)

  def __getitem__(self, idx: int):
    image_id = self.image_ids[idx]
    image = cv2.imread(self.images[idx], cv2.IMREAD_UNCHANGED)
    # image = cv2.imread(self.images[idx])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if os.path.exists(self.masks[idx]):
      mask = cv2.imread(self.masks[idx], cv2.IMREAD_UNCHANGED)
      # mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
    else:
      mask = None

    # apply augmentations
    if self.augmentation:
      sample = self.augmentation(image=image, mask=mask)
      image, mask = sample['image'], sample['mask']

    # apply preprocessing
    if self.preprocessing:
      sample = self.preprocessing(image=image, mask=mask)
      image, mask = sample['image'], sample['mask']

    x = {
      'image': image,
      'id': image_id
    }
    y = mask

    return x, y

  def __len__(self) -> int:
    return len(self.image_ids)
  
