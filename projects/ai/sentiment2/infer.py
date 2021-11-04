#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   train.py
#        \author   chenghuige  
#          \date   2018-01-13 16:32:26.966279
#   \Description  
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

import tensorflow as tf 
flags = tf.app.flags
FLAGS = flags.FLAGS
  
tf.enable_eager_execution()

import numpy as np
from tqdm import tqdm

import melt 
logging = melt.logging
import gezi
import traceback

from wenzheng.utils import input_flags 

#from algos.model import *
from algos.loss import criterion
import algos.model as base
from dataset import Dataset
import evaluate as ev

from prepare.text2ids import text2ids
from wenzheng.utils import ids2text
import numpy as np

from algos.config import ATTRIBUTES


def main(_):
  melt.apps.init()
  
  #ev.init()

  model = getattr(base, FLAGS.model)()
  model.debug = True

  melt.eager.restore(model)

  ids2text.init()
  vocab = ids2text.vocab

  content = '这是一个很好的餐馆，菜很不好吃，我还想再去'
  content = '这是一个很差的餐馆，菜很不好吃，我不想再去'
  content = '这是一个很好的餐馆，菜很好吃，我还想再去'
  content = '这是一个很好的餐馆，只是菜很难吃，我还想再去'
  content = '这是一个很好的餐馆，只是菜很不好吃，我还想再去'

  cids = text2ids(content)
  words = [vocab.key(cid) for cid in cids]
  print(cids)
  print(ids2text.ids2text(cids))
  x = {'content': [cids]}
  logits = model(x)[0]
  probs = gezi.softmax(logits, 1)
  print(probs)
  print(list(zip(ATTRIBUTES, [list(x) for x in probs])))

  predicts = np.argmax(logits, -1) - 2
  print('predicts ', predicts)
  print(list(zip(ATTRIBUTES, predicts)))
  adjusted_predicts = ev.to_predict(logits)
  print('apredicts', adjusted_predicts)
  print(list(zip(ATTRIBUTES, adjusted_predicts)))

  # print words importance scores
  word_scores_list = model.pooling.word_scores

  for word_scores in word_scores_list:
    print(list(zip(words, word_scores[0].numpy())))


if __name__ == '__main__':
  tf.app.run()  
