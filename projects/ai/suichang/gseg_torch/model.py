#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   model.py
#        \author   chenghuige  
#          \date   2020-10-11 13:04:23.334055
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import torch
from torch import nn

# import segmentation_models_pytorch as smp
# from segmentation_models_pytorch.encoders import get_preprocessing_fn
from .third import segmentation_models_pytorch as smp
# from smp.encoders import get_preprocessing_fn

from gseg.config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
  def __init__(self, num_classes, model_name='Unet', backbone='resnet50', 
               activation='softmax'):
    super(Model, self).__init__() 
    MyModel = getattr(smp, model_name)
    kwargs = {}
    if model_name == 'Unet':
      kwargs ={
          'decoder_channels': (128, 64, 32, 16, 8),
          'decoder_attention_type': None if not FLAGS.use_scse else 'scse',
      }
    self.model = MyModel(backbone, classes=num_classes, encoder_weights='imagenet', 
                         activation=activation, in_channels=3,
                         **kwargs
                         )
    self.preprocess = smp.encoders.get_preprocessing_fn(backbone, pretrained='imagenet')

  def forward(self, input):
    if not FLAGS.torch_only:
      # device = gezi.get('device') or device
      # 这个多gpu会挂
      # x = self.preprocess(input['image'].cpu()).to(device)
      # 这个无法cpu
      x = self.preprocess(input['image'].cpu()).cuda()
      # 内部有.numpy得先转cpu运算
      # x = self.preprocess(input['image'])
      # https://github.com/pytorch/pytorch/issues/42300
      x = x.permute(0, 3, 1, 2).contiguous().to(dtype=torch.float32)
    else:
      x = input['image']
      
    y = self.model(x)
    return y

def get_model():
  model_name = ''.join(FLAGS.model.split('.')[1:]) if '.' in FLAGS.model else FLAGS.model
  return Model(FLAGS.NUM_CLASSES, model_name=model_name, backbone=FLAGS.backbone)
