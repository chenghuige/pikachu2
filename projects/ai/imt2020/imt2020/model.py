#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   model.py
#        \author   chenghuige  
#          \date   2021-01-09 17:51:25.245765
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import tensorflow as tf
from tensorflow import keras
from transformers import TFAutoModel, BertConfig

import melt as mt
from .config import *

from .modelDesign import *

def get_model(model_name=None):
  encInput = keras.Input(shape=(CHANNEL_SHAPE_DIM1, CHANNEL_SHAPE_DIM2, CHANNEL_SHAPE_DIM3))
  encOutput = Encoder(encInput, NUM_FEEDBACK_BITS)
  encModel = keras.Model(inputs=encInput, outputs=encOutput, name='Encoder')
  # Decoder
  decInput = keras.Input(shape=(NUM_FEEDBACK_BITS,))
  decOutput = Decoder(decInput, NUM_FEEDBACK_BITS)
  decModel = keras.Model(inputs=decInput, outputs=decOutput, name="Decoder")
  # Autoencoder
  autoencoderInput = keras.Input(shape=(CHANNEL_SHAPE_DIM1, CHANNEL_SHAPE_DIM2, CHANNEL_SHAPE_DIM3))
  autoencoderOutput = decModel(encModel(autoencoderInput))
  autoencoderModel = keras.Model(inputs=autoencoderInput, outputs=autoencoderOutput, name='Autoencoder')
  return autoencoderModel, encModel, decModel
