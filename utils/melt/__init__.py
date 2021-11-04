from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
warnings.simplefilter("ignore")

# try:
#   import matplotlib
#   matplotlib.use('Agg')
# except Exception:
#   pass

import tensorflow as tf 
# https://github.com/tensorflow/tensorflow/issues/27023 remove depreciated warning
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import traceback

import sys
# print('tensorflow_version:', tf.__version__, file=sys.stderr) 

try:
  import torch
  # print('torch_version:', torch.__version__, file=sys.stderr) 
except Exception:
	print("torch not found", file=sys.stderr)

from melt.training import training as train 
import melt.training 

import melt.utils
import gezi 
from gezi import logging
from melt.utils import EmbeddingSim

from melt.util import *
from melt.ops import *
from melt.variable import * 
from melt.tfrecords import * 
from melt.tfrecords.dataset import Dataset

from melt.inference import *

import melt.layers
from melt.layers.layers import activation_layer as activation

# import melt.slim2

import melt.flow
# from melt.flow import projector_config

from melt.metrics import * # TODO

try:
  import melt.apps
  from melt.apps.init import *
  from melt.apps.train import *
except Exception:
	print(traceback.format_exc(), file=sys.stderr)
	pass

import melt.rnn 
# import melt.cnn 
import melt.encoder 

import melt.seq2seq 
import melt.image  
from melt.image import *

import melt.losses  

import melt.eager 

import melt.distributed 
from melt.distributed import get_strategy

import melt.deepctr
import melt.global_objectives

import melt.models
import melt.pretrain

# import melt.torch 
