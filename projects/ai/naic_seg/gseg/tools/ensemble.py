#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   ensemble.py
#        \author   chenghuige  
#          \date   2020-11-25 11:45:33.320942
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
sys.path.append('..')

import glob
import numpy as np
import cv2
import tensorflow as tf
import gezi
from gezi import tqdm
from gezi.metrics.image.semantic_seg import Evaluator
import melt as mt
from gseg.dataset import Dataset
from gseg.loss import get_loss_fn
from gseg.metrics import get_metrics
from gseg import util  

CLASSES = ['water', 'track_road', 'build', 'track_airport', 'other_park', 'other_playground', 'arable_natural', 'arable_greenhouse',
           'grass_natural', 'grass_greenbelt', 'forest_natural', 'forest_planted', 'bare_natural', 'bare_planted', 'other_other']
NUM_CLASSES = len(CLASSES)

def _predict(model, imgs, tta=False):
  if not tta:
    pred = model.predict(imgs)
  else:
    imgs_list = [imgs, tf.image.flip_left_right(imgs), tf.image.flip_up_down(imgs)]
    imgs = tf.concat(imgs_list, axis=0)
    res = np.split(model.predict(imgs), len(imgs_list))
    res[1] = tf.image.flip_left_right(res[1]).numpy()
    res[2] = tf.image.flip_up_down(res[2]).numpy()
    # pred = np.mean(res, axis=0)
    pred = (res[0] + res[1] + res[2]) / 3.
  return pred
    
def predicts(model, indir='../input/eval.naic/image', odir='../input/out', batch_size=4, tta=False, num_imgs=0):
  gezi.try_mkdir(odir)
  os.system(f'rm -rf {odir}/*')
  img_paths = glob.glob(f'{indir}/*')
  num_imgs = num_imgs or len(img_paths)
  for i in tqdm(range(0, num_imgs, batch_size), desc='predict', ascii=True, leave=True):
    begin = i
    end = min(i + batch_size, num_imgs)
    imgs = np.asarray([gezi.imread(img_path) for img_path in img_paths[begin:end]])
    logits = _predict(model, imgs, tta)
    preds = logits.argmax(-1)
    for img_path, pred in zip(img_paths[begin:end], preds):
      index = os.path.basename(img_path).split('.')[0]
      cv2.imwrite(os.path.join(odir, f'{index}.png'), pred)
      
def predicts_eval(model, indir='../input/eval.naic/image', odir='../input/out', batch_size=4, tta=False, num_imgs=0):
  m = {}
  for i in range(17):
    if i < 4:
      m[i + 1] = i
    else:
      m[i + 1] = i - 2

  gezi.try_mkdir(odir)
  os.system(f'rm -rf {odir}/*')
  img_paths = glob.glob(f'{indir}/*')
  label_dir = indir.replace('image', 'label')
  num_imgs = num_imgs or len(img_paths)
  evaluator = Evaluator(CLASSES)
  key_metric = 'FWIoU'
  class_mious = 0.
  t = tqdm(range(0, num_imgs, batch_size), desc='predict', ascii=True, leave=True)
  for i in t:
    begin = i
    end = min(i + batch_size, num_imgs)
    imgs = np.asarray([gezi.imread(img_path) for img_path in img_paths[begin:end]])
    logits = _predict(model, imgs, tta)
    preds = logits.argmax(-1)
    for img_path, pred in zip(img_paths[begin:end], preds):
      index = os.path.basename(img_path).split('.')[0]
      cv2.imwrite(os.path.join(odir, f'{index}.png'), pred)
      
      label_file = f'{label_dir}/{index}.png'
      label = cv2.imread(label_file, cv2.IMREAD_UNCHANGED).astype(np.uint8)
      f = np.vectorize(lambda x: m[x])
      label = f(label).astype(np.uint8)
      evaluator.eval_each(label[np.newaxis,:, :], pred[np.newaxis,:, :], metric=key_metric)
      binary_label = np.bincount(label.reshape(-1), minlength=NUM_CLASSES).astype(np.bool)
      binary_pred = np.bincount(pred.reshape(-1), minlength=NUM_CLASSES).astype(np.bool)
      intersections = binary_label * binary_pred
      intersection = np.sum(intersections, axis=-1)
      union = np.sum(binary_label, axis=-1) + np.sum(binary_pred, axis=-1) - intersection
      class_miou = np.mean(intersection / union)
      class_mious += class_miou
      t.set_postfix({key_metric: evaluator.eval_once(key_metric), 'IMAGE/CLASS/MIoU': class_mious / (i + 1)})

  res = evaluator.eval_once()
  res['IMAGE/CLASS/MIoU'] = class_mious / i
  gezi.set('evaluator', evaluator)
  return res

mt.init_flags()
FLAGS = mt.get_flags()
FLAGS.batch_parse = False
batch_size = 32
batch_size = 16
batch_size = 8
mt.set_global('batch_size', batch_size) # loss fn used / mt.batch_size()
eval_files = gezi.list_files('../input/quarter/tfrecords/train/1/*')
eval_dataset = Dataset('valid').make_batch(batch_size, eval_files)
train_files = gezi.list_files('../input/quarter/tfrecords/train/*/*')
train_files = [x for x in train_files if not x in eval_files]
train_dataset = Dataset('train').make_batch(batch_size, train_files)
train_steps = -(-mt.get_num_records(train_files) // batch_size)
steps = -(-mt.get_num_records(eval_files) // batch_size)
FLAGS.NUM_CLASSES = NUM_CLASSES

def eval(model):
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  metrics = get_metrics()
  model.compile('sgd', loss_fn, metrics=metrics)
  res = model.evaluate(eval_dataset, steps=steps, return_dict=True)
  gezi.pprint_dict(res)
  cm = mt.distributed.sum_merge(metrics[0].get_cm())
  infos = util.get_infos_from_cm(cm, CLASSES)
  res.update(infos)
  return res

root = '../working/convert'
dest = '../working/ensemble'

if len(sys.argv) > 1:
  model_paths = sys.argv[1].split(',')
else:
  model_paths = glob.glob(f'{root}/*')

bin_names = ['model.h5', 'model_lr.h5', 'model_ud.h5', 'model_rot1.h5', 'model_rot2.h5', 'model_rot3.h5']

def get_model_path(i):
  model_path = model_paths[i]
  bin_name = bin_names[i % len(bin_names)]
  if not os.path.exists(model_path):
    model_path = os.path.join(root, model_path)
  model_path = os.path.join(model_path, bin_name)
  return model_path

#info_file = os.path.join(dest, 'ensemble.txt')
model_paths = [get_model_path(i) for i in range(len(model_paths))]
#model_paths_str = '|'.join(model_paths)

#for line in open(info_file):
#  info = line.strip().split(',')[-1]
#  if model_paths_str == info:
#    print('Already done:', line.strip())
#    exit(0)

print(model_paths)
model = mt.EnsembleModel(model_paths).get_model()

res = eval(model)
metric_name = 'FWIoU'
metric = res[metric_name]
print(f'{metric_name}: {metric}') 

#info = f'{metric},{model_paths_str}'
#gezi.append_to_txt(info, info_file)

model_path = os.path.join(dest, 'model.h5')
mt.save_model(model, model_path)
#gezi.write_to_txt(info, model_path.replace('model.h5', 'ensemble.txt'))

