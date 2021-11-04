#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   post_process.py
#        \author   chenghuige  
#          \date   2020-10-11 14:06:37.946000
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
  
import numpy as np

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels

from skimage.morphology import remove_small_holes, remove_small_objects

# Fully connected CRF post processing function
def do_crf(image, mask, zero_unsure=True):
    colors, labels = np.unique(mask, return_inverse=True)
    image_size = mask.shape[:2]
    n_labels = len(set(labels.flat))

    if n_labels == 1:
        return mask

    d = dcrf.DenseCRF2D(image_size[1], image_size[0], n_labels)  # width, height, nlabels
    U = unary_from_labels(labels, n_labels, gt_prob=.7, zero_unsure=zero_unsure)
    d.setUnaryEnergy(U)
    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(3,3), compat=3)
    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    # im is an image-array, e.g. im.dtype == np.uint8 and im.shape == (640,480,3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image.astype('uint8'), compat=10)
    Q = d.inference(5) # 5 - num of iterations
    MAP = np.argmax(Q, axis=0).reshape(image_size)
    unique_map = np.unique(MAP)
    for u in unique_map: # get original labels back
        np.putmask(MAP, MAP == u, colors[u])
    return MAP
    # MAP = do_crf(frame, labels.astype('int32'), zero_unsure=False)

def remove_small_objects_and_holes(mask, num_classes, min_size=30, area_threshold=30):
    masks = []
    # print(mask)
    # print(len(np.unique(mask)))
    for i in range(num_classes):
        # print('-----1', i, np.sum((mask == i).astype(np.int32)) / 65536)
        mask_ = remove_small_objects(mask==i,min_size=min_size,connectivity=1)
        mask_ = remove_small_holes(mask_==1,area_threshold=area_threshold,connectivity=1) 
        # print('-----2', i, np.sum((mask_ == 1).astype(np.int32)) / 65536)
        masks.append(mask_)
    mask = np.stack(masks, -1).astype(np.float32)       
    return mask
