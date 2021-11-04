#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   util.py
#        \author   chenghuige  
#          \date   2020-09-28 19:54:22.741818
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import tensorflow as tf
import numpy as np
import cv2

import zipfile
import matplotlib.pyplot as plt
import io

import gezi
from gezi import tqdm, plot, logging
import melt as mt
from melt.distributed import tonumpy

from .config import *
try:
  from .post_processing import *
except Exception as e:
  pass

def get_infos_from_cm(cm, CLASSES):
  NUM_CLASSES = cm.shape[0]
  infos = {}
  # move to eval_image 
  trues = np.sum(cm, 1)
  preds = np.sum(cm, 0)
  total = np.sum(trues)
  classes_ratio_true = trues / total
  classes_ratio_pred = preds / total
  for i in range(len(CLASSES)):
    infos[f'PIXEL/true/{CLASSES[i]}'] = classes_ratio_true[i] 
    infos[f'PIXEL/pred/{CLASSES[i]}'] = classes_ratio_pred[i] 
    iou = cm[i][i] / (trues[i] + preds[i] - cm[i][i])
    fwiou = 1. - (1. - iou) * classes_ratio_true[i] * NUM_CLASSES
    acc = cm[i][i] / preds[i]
    recall = cm[i][i] / trues[i]
    print('{:20s}'.format(CLASSES[i]), '|true:{:.4f} pred:{:.4f} acc:{:.4f} recall:{:.4f} iou:{:.4f} fwiou:{:.4f}|'.format(classes_ratio_true[i], classes_ratio_pred[i], acc, recall, iou, fwiou))
    infos[f'IoU/{CLASSES[i]}'] = iou
    infos[f'FWIoU/{CLASSES[i]}'] = fwiou
    infos[f'ACC/{CLASSES[i]}'] = acc
    infos[f'REC/{CLASSES[i]}'] = recall
  return infos

def is_classifier():
  return FLAGS.model.startswith('classifier')

def pixel2class_label(y, input=None):
  NUM_CLASSES = FLAGS.NUM_CLASSES
  pixels = FLAGS.ori_image_size[0] * FLAGS.ori_image_size[1]
  if input is None:
    bs = mt.get_shape(y, 0)
    # why return shape <unkown> .. tpu not ok
    y = tf.math.bincount(tf.reshape(y, (-1, pixels)), minlength=NUM_CLASSES, maxlength=NUM_CLASSES, axis=-1, binary_output=True)
    y.set_shape((bs, NUM_CLASSES))
    # print(y.shape)
  else:
    y = tf.cast(input['bins'] > 0, tf.int32)
  return y

# 2020-10-15 15:39:26 2:42:36 water    |true:0.1110 pred:0.1082 acc:0.9453 recall:0.9215 iou:0.8748 fwiou:0.8888|
# 2020-10-15 15:39:26 2:42:36 track    |true:0.0625 pred:0.0633 acc:0.8274 recall:0.8378 iou:0.7131 fwiou:0.8565|
# 2020-10-15 15:39:26 2:42:36 build    |true:0.1845 pred:0.1827 acc:0.9239 recall:0.9147 iou:0.8506 fwiou:0.7795|
# 2020-10-15 15:39:26 2:42:36 arable   |true:0.1635 pred:0.1653 acc:0.8439 recall:0.8533 iou:0.7370 fwiou:0.6559|
# 2020-10-15 15:39:26 2:42:36 grass    |true:0.0855 pred:0.0849 acc:0.8412 recall:0.8347 iou:0.7211 fwiou:0.8091|
# 2020-10-15 15:39:26 2:42:36 forest   |true:0.1558 pred:0.1540 acc:0.9266 recall:0.9161 iou:0.8541 fwiou:0.8182|
# 2020-10-15 15:39:26 2:42:36 bare     |true:0.0498 pred:0.0487 acc:0.8498 recall:0.8317 iou:0.7251 fwiou:0.8905|
# 2020-10-15 15:39:26 2:42:36 other    |true:0.1873 pred:0.1928 acc:0.8119 recall:0.8359 iou:0.7003 fwiou:0.5510|

# # lower score..
# def adjust_preds(preds):
#   if FLAGS.from_logits:
#     preds = gezi.softmax(preds)
#   class_ratios = [0.111, 0.0625, 0.1845, 0.1635, 0.0855, 0.1558, 0.0487, 0.1873]
#   weights = np.asarray(class_ratios)
#   # preds *= weights
#   preds /= weights
#   return preds

def to_pred(preds, image=None):
  # if FLAGS.adjust_preds:
  #   preds = adjust_preds(preds)
  preds = np.argmax(preds, axis=-1)
  preds = preds.astype(np.uint8)
  # preds = post_deal(preds, y, image)
  if FLAGS.interpolation:
    preds = np.concatenate([cv2.resize(pred, tuple(FLAGS.ori_image_size), interpolation=cv2.INTER_NEAREST)[np.newaxis,:,:] for pred in preds], axis=0)
  return preds

def post_deal(masks, probs=None, images=None):
  NUM_CLASSES = FLAGS.NUM_CLASSES
  if FLAGS.post_crf:
    masks_ = []
    for image, mask in zip(images, masks):
      mask = do_crf(image, mask, zero_unsure=True)
      masks_.append(mask)
    masks = np.stack(masks_, axis=0)
  if FLAGS.post_remove:
    masks_ = []
    for mask, prob in zip(masks, probs):
      # print('------1', len(np.unique(mask)))
      mask = remove_small_objects_and_holes(mask, NUM_CLASSES, FLAGS.min_size, FLAGS.min_size)
      # print('-----2', len(np.unique(mask)))
      prob *= mask
      mask = np.argmax(prob, axis=-1)
      mask = remove_small_objects_and_holes(mask, NUM_CLASSES, FLAGS.min_size, FLAGS.min_size)
      prob *= mask
      mask = np.argmax(prob, axis=-1)
      masks_.append(mask)
    masks = np.stack(masks_, axis=0)
  return masks

m = {}
for i in range(15):
  if i < 4:
    m[i] = i + 1
  else:
    m[i] = i + 3

def to_submit(pred):
  if FLAGS.data_version == 1:
    return ((pred.astype(np.uint16) + 1) * 100).astype(np.uint16)
  elif FLAGS.data_version == 2:
    f = np.vectorize(lambda x: m[x])
    pred = f(pred).astype(np.uint8)
    return pred
  else:
    raise ValueError(FLAGS.data_version)

# def write_zipped_results(ids, predicts, outdir):
#   oname = outdir + '/results.zip'
#   with zipfile.ZipFile(oname, "w", compression=zipfile.ZIP_DEFLATED) as zf:
#     for id, pred in zip(ids, predicts):
#       pred = to_submit(pred)
#       plt.imshow(pred)
#       buf = io.BytesIO()
#       plt.savefig(buf, format='png')
#       plt.close()
#       buf = io.BytesIO()
#       img_name = f'{outdir}/{id}.png'
#       zf.writestr(img_name, buf.getvalue())

def zip_files(outdir):
  # gezi.make_archive(f'{rootdir}/results', 'zip', rootdir, 'results')
  os.system(f'cd {outdir};zip -qq results.zip results/*')

def write_tfrec_results(ids, predicts, outdir, mode, masks=None):
  # TODO FIXME valid 和 test会混在一起 需要分开 tfrec_valid_writer tfrecord_test_writer
  writer = gezi.get(f'tfrec_{mode}_writer')
  if not writer:
    writer = gezi.set(f'tfrec_{mode}_writer', mt.tfrecords.MultiWriter(dir=outdir, max_records=250))
  
  if masks is None:
    masks = [None] * len(ids)
  for id, pred, mask in zip(ids, predicts, masks):
    feature = {}
    feature['id'] = id
    feature['pred'] = pred.reshape(-1).astype(np.float16).tobytes()
    if mask:
      feature['mask'] = mask.reshape(-1).astype(np.uint16).tobytes()
    writer.write_feature(feature)

def write_results(ids, predicts, outdir, mode):
  if mode == 'test':
    outdir += '/results'
  else:
    outdir += '/eval_results'
  os.makedirs(outdir, exist_ok=True)
  # for id, pred in tqdm(zip(ids, predicts), total=len(ids), desc='write'):
  for id, pred in zip(ids, predicts):
    pred = to_submit(pred)
    cv2.imwrite(outdir + f'/{id}.png', pred)

def write_inter_results(ids, predicts, outdir, mode, out_type='npy', masks=None):
  if mode == 'test':
    outdir += '/test_inter_results'
  else:
    outdir += '/eval_inter_results'
  os.makedirs(outdir, exist_ok=True)

  if out_type == 'npy':
    # # for valid ensemble also test ensemble
    for id, pred in zip(ids, predicts):
      pred = tf.quantization.quantize(pred, 0, 1, tf.quint8).output.numpy()
      np.save(outdir + f'/{id}.npy', pred.astype(np.uint8))
  elif out_type == 'tfrec':
    write_tfrec_results(ids, predicts, outdir, mode)
  else:
    raise ValueError(out_type)

# # 直接写zip提交结果 TODO 配合valid ensemble 都直接写tfrecord results路径下面是tfrecord
# def write_infer(ids, predicts, outdir):
#   if not FLAGS.write_test_image:
#     return
#   write_results(ids, predicts, outdir)
#   # write_zipped_results(ids, predicts, outdir)
#   # write_tfrec_results(ids, predicts, outdir)

# def write_valid(ids, labels, predicts, outdir):
#   if not FLAGS.write_valid_image:
#     return
#   write_results(ids, predicts, outdir, is_test=False)
#   # write_zipped_results(ids, predicts, outdir)
#   # write_tfrec_results(ids, predicts, outdir)

## 多gpu用strategy.run能跑 多个gpu确实都在运算 但是没有性能收益.. 如果不用strategy.run只有一个gpu在跑 速度也差不都一样 
def inference(dataset, model, steps, num_examples, outdir, desc='inference'):
  CLASSES = FLAGS.CLASSES
  NUM_CLASSES = FLAGS.NUM_CLASSES

  # prun.py
  if FLAGS.parts:
    FLAGS.zip_image = False
    FLAGS.show_imgs = int(FLAGS.show_imgs / FLAGS.parts)
    FLAGS.show_rand_imgs = int(FLAGS.show_rand_imgs / FLAGS.parts)
    
  @tf.function
  def _infer_step(x):
    y_ = model(x, training=False)
    return y_

  logger = mt.get_summary_writer()

  indexes = np.asarray(range(num_examples))
  np.random.seed(FLAGS.img_show_seed)
  np.random.shuffle(indexes)
  indexes = indexes[:FLAGS.show_rand_imgs]
  random_imgs = []

  def _tb_image(imgs, tag, i):
    id = str(imgs[0])
    img = imgs[1]
    text = id
    pred_ = imgs[2]
    prob = imgs[3]
    step = 1
    # logger.images(f'{tag}_test/{i}', [img, pred_], step, title=id)
    img_ = plot.segmentation(img, pred_, CLASSES, prob=prob, title=id)
    logger.image(f'{tag}_test/{i}', img_, step) 

  strategy = mt.distributed.get_strategy()
  dataset = strategy.experimental_distribute_dataset(dataset)
  cur_step = 0
  for step_, (x, _) in tqdm(enumerate(dataset), total=steps, ascii=True, desc=desc):
    if step_ == steps:
      break
    ## 不适合多卡，而且多卡只有一个在运算
    # y_ = model(x)
    ## 速度更慢... 
    # y_ = model.predict_on_batch(x)
    ## 多卡多个运算 虽然也没有加速比 但是整体兼容性最好 
    y_ = strategy.run(_infer_step, args=(x,))
    
    ids = x['id']
    imgs = x['image']
    # ids, imgs = ids.numpy(), imgs.numpy()
    # y_, ids, imgs = y_.numpy(), ids.numpy(), imgs.numpy()
    y_, ids, imgs = tonumpy(y_, ids, imgs)

    imgs /= 255.
    
    ids = gezi.squeeze(ids)
    preds = to_pred(y_)
    probs = gezi.softmax(y_, axis=-1)
    prob = np.max(probs, -1)

    if FLAGS.write_inter_results:
      # write_inter_results(ids, y_, outdir, mode='test')
      write_inter_results(ids, probs, outdir, mode='test')

    if FLAGS.write_test_results and not FLAGS.write_inter_results:
      write_results(ids, preds, outdir, mode='test')

    if FLAGS.tb_image:
      for i in range(len(ids)):
        if cur_step in indexes:
          item = [ids[i], imgs[i], preds[i], prob[i]]
          # random_imgs += [item]
          index = list(indexes).index(cur_step)
          _tb_image(item, 'TestImages', index)
        
        cur_step += 1
  
  if FLAGS.write_test_results:
    if FLAGS.zip_image:
      zip_files(outdir) 

  writer = gezi.get('tfrec_test_writer')
  if writer:
    writer.close()

# 似乎也没啥意义 加速比没有 model.infer切换数据时间代价过高 似乎最快的方案还是单独test模式指定不同数据 多个并行启动test了
# 但是tpu环境还不知道如何指定使用某个tpu core ...
def fast_inference(dataset, model, steps, ofile): 
  from .dataset import Dataset
  # def infer_step(model, x, y_):
  #   ids = x['id']
  #   ids = ids.numpy()
  #   ids = gezi.squeeze(ids)
  #   preds = to_pred(y_)
  #   write_infer(ids, preds, ofile)
  ## not work as cutom_loop cant not use .numpy.. but if you do not need numpy you can use this
  # mt.custom_loop(model, dataset, infer_step, steps=steps, mark='predict')

  files = gezi.list_files(FLAGS.test_input)
  # for file in tqdm(files, ascii=True, desc='test_datasets'):
  for i in tqdm(range(FLAGS.fast_infer_steps), ascii=True, desc='test_datasets'):
    start, end = gezi.get_fold(len(files), FLAGS.fast_infer_steps, i)
    ds = Dataset('test')
    # dataset = ds.make_batch(melt.eval_batch_size(), [file])
    dataset = ds.make_batch(mt.eval_batch_size(), files[start:end])
    file = files[0]
    steps = -(-len(ds) // mt.eval_batch_size())
    outputs = model.infer(dataset, steps=steps, desc=f'test_predict:{file}', verbose=1)
    y_ = outputs['pred']
    ids = outputs['id']

    ids = gezi.squeeze(ids)
    preds = to_pred(y_)
    # 写文件也需要一定时间代价
    write_infer(ids, preds, ofile)
  rootdir = os.path.dirname(ofile) 
  zip_files()

def get_infer_fn():
  if FLAGS.fast_infer:
    # fast inference will OOM even cpu just use inference
    # return None
    return fast_inference
  else:
    return inference

def get_teacher():
  teacher =gezi.get('teacher')
  if teacher:
    return teacher
  else:
    if FLAGS.teacher:
      teacher_path = FLAGS.teacher if not os.path.isdir(FLAGS.teacher) else os.path.join(FLAGS.teacher, 'model.h5')
      with gezi.Timer(f'Loading teacher: {teacher_path}', print_fn=logging.info, print_before=True):
        strategy = mt.distributed.get_strategy()
        with strategy.scope():
          teacher = mt.load_model(teacher_path)
          teacher.trainable = False
      gezi.set('teacher', teacher)
    return teacher

# --------------------  final metrics
def get_size(x):
  size = os.path.getsize(x) / 1024. / 1024.
#   print(size)
  command = f'rm -rf /tmp/x.zip;zip /tmp/x.zip {x}'
#   print(command)
  os.system(command)
  size = os.path.getsize('/tmp/x.zip') / 1024. / 1024.
#   print(size)
  return size

def get_size_score(size):
  if size >= 50:
    return 0 
  else:
    return 0.2 * (1. - size / 50)

def get_perf_score(hours):
  if hours >= 6:
    return 0
  else:
    return 0.3 * (1. - hours / 6)

def get_metric_score(fwiou):
  # 0.787 -> 0.581
  return 0.5 * (fwiou - 0.206)

def get_score(size, duration, fwiou):
  return get_size_score(size) + get_perf_score(duration) + get_metric_score(fwiou)

def scoring(model_path, eval_fn, model=None, size=None, base_dur=276, dur=None, fwiou=None, size_div=2.):
  if model is None:
    model = mt.load_model(model_path)
  # 模型可以通过fp16简单压缩一倍
  if size is None:
    size = os.path.getsize(model_path) / 1024. / 1024. / size_div
  size_score = get_size_score(size)
  if not size_score:
    return size
  if dur is None or fwiou is None:
    timer = gezi.Timer() 
    ret = eval_fn(model)
    fwiou = ret['FWIoU']
    dur = timer.elapsed()
  hours = dur * 6 / base_dur
  perf_score = get_perf_score(hours)
  metric_score = get_metric_score(fwiou)
  other_score = size_score + perf_score
  score = metric_score + other_score
  res = {
    'size': size,
    'size_score': size_score,
    'dur': dur,
    'hours': hours,
    'perf_score': perf_score,
    'FWIoU': fwiou,
    'metric_score': metric_score,
    'other_score': other_score,
    'score': score
  }
  return res
