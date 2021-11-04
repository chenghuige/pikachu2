#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   __init__.py
#        \author   chenghuige  
#          \date   2016-09-26 20:28:47.577482
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from absl import flags
FLAGS = flags.FLAGS

import melt.metrics.rank_metrics 
import melt.metrics.metrics  # TODO how to import many bu except like keras.layer..
