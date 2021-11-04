#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   ensemble.py
#        \author   chenghuige  
#          \date   2020-10-17 15:59:03.736408
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import glob
import numpy as np
import cv2

import tensorflow as tf

import gezi
from gezi import logging, tqdm
import melt as mt

from .evaluate import ImageEvaluator
from .util import *
from .config import *

class EnsembleDataset(mt.Dataset):
  def __init__(self, subset='valid', **kwargs):
    super(EnsembleDataset, self).__init__(subset, **kwargs)

  def parse(self, example):
    features_dict = {
      'id':  tf.io.FixedLenFeature([], tf.int64),
      'pred': tf.io.FixedLenFeature([], tf.string)
    } 
    if self.subset == 'valid':
      features_dict['mask'] = tf.io.FixedLenFeature([], tf.string)

    f = self.parse_(serialized=example, features=features_dict)

    f['pred'] = tf.reshape(tf.io.decode_raw(f['pred'], tf.float16), (*FLAGS.image_size, FLAGS.NUM_CLASSES))
    f['pred'] = tf.cast(f['pred'], tf.float32)

    if self.subset == 'valid':
      f['mask'] = tf.reshape(tf.io.decode_raw(f['pred'], tf.uint16), (*FLAGS.image_size))
    else:
      f['mask'] = tf.zeros_like(f['id'])

    x = f
    y = f['mask']
    del f['mask']

    return x, y

def ensemble_tfrec(dirs, mode='test', outdir=None, num_images=None, display_results=True):
  batch_size = 1024
  datasets = []
  ds = EnsembleDataset(mode)
  for dir_ in dirs:
    print(dir_)
    files = glob.glob(f'{dir_}/*.tfrec')
    files = [x for x in sorted(files, key=os.path.getmtime)]
    print(files)
    datasets.append(iter(ds.make_batch(batch_size, files, repeat=False, drop_remainder=False, shuffle=False)))

  # ## TODO test dataset顺序如何和ensemble 保持一致 方便展示效果 ？
  # # test_dataset = iter(gezi.get('info')['test_dataset'])
  # test_files = gezi.list_files(FLAGS.test_input)
  # test_dataset = iter(Dataset('test').make_batch(mt.eval_batch_size(), repeat=False, drop_remainder=False, shuffle=False))

  num_datas = len(datasets)

  logging.info(f'ensemble {num_datas} models')

  num_examples = len(ds)

  logging.info('num_examples:', num_examples)
  num_steps = -(-num_examples // batch_size)
  logging.info('num_steps:', num_steps)

  for step in tqdm(range(num_steps)):
    try:
      preds = []
      for i in range(num_datas):
        x, _ = next(datasets[i])
        # print(x['id'])
        # print(x['pred'][0][0])
        # print(to_pred(x['pred'])[0])
        preds.append(x['pred'])

      # x_, _ = next(test_dataset)
      # print(x_['id'])
      preds = tf.reduce_mean(tf.stack(preds, axis=0), axis=0).numpy()
      # print(preds[0][0])
      ids = x['id'].numpy()
      preds = to_pred(preds)
      # print(preds[0])
      write_results(ids, preds, outdir, mode)

      # probs = gezi.softmax(preds, axis=-1)
      # prob = np.max(probs, -1)
      
      # if step % 20 == 0:
      #   item = [ids[i], imgs[i], preds[i], prob[i]]
      #   # random_imgs += [item]
      #   index = list(indexes).index(cur_step)
      #   _tb_image(item, '4Random', index)

    except Exception as e:
      print(e)
      pass
    # break

# ensemble from numpy files
def ensemble_npy(dirs, label_dir=None, outdir=None, num_images=None, display_results=True):
  key_metric = 'FWIoU'
  res = {}
  dir_ = dirs[0]
  mode = 'valid' if label_dir else 'test'
  files = glob.glob(f'{dir_}/*.npy')
  image_ids = [os.path.basename(file).split('.')[0] for file in files]
  if num_images:
    image_ids = image_ids[:num_images]
  logging.info('Num image ids:', len(image_ids))
  if mode == 'valid':
    image_dir = os.path.dirname(label_dir) + '/image'
    evaluator = ImageEvaluator(len(image_ids), display_results=display_results)
  else:
    assert out_dir
    out_dir = out_dir + '/results'
    gezi.try_mkdir(out_dir)

  t = tqdm(image_ids, total=len(image_ids), desc=f'ensemble_{mode}', ascii=True)
  for image_id in t:
    if mode == 'valid':
      label = cv2.imread(f'{label_dir}/{image_id}.png', cv2.IMREAD_UNCHANGED)
      label = label / 100 - 1
      image = cv2.imread(f'{image_dir}/{image_id}.tif', cv2.IMREAD_UNCHANGED)
    
    preds = []
    for dir_ in dirs:
      preds.append(np.load(f'{dir_}/{image_id}.npy').astype(np.float32))
    
    preds = np.stack(preds, axis=0)
    pred = np.mean(preds, axis=0)

    if mode == 'valid':
      evaluator(image_id, image, label, pred)
      res = evaluator.eval_once()
      t.set_postfix({key_metric: res[key_metric]})
    else:
      mask = np.mean(pred, axis=-1)
      mask = to_submit(mask)
      cv2.imwrite(outdir + f'/{id}.png', mask)
    
  if mode == 'valid':
    res.update(evaluator.eval_once())
    gezi.pprint_dict(res)
    evaluator.finalize()
  
