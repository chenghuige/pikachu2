#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   pca.py
#        \author   chenghuige  
#          \date   2021-07-25 01:13:32.177647
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import numpy as np
# TODO
#sklearn/decomposition/_pca.py:561: RuntimeWarning: invalid value encountered in true_divide
#  self.explained_variance_ / total_var.sum()
np.seterr(divide='ignore',invalid='ignore')
from sklearn.decomposition import PCA

dim = int(sys.argv[3])

x = np.load(sys.argv[1])
pca = PCA(n_components=dim)
pca.fit(x)
x = pca.transform(x)
np.save(sys.argv[2], x)
  
