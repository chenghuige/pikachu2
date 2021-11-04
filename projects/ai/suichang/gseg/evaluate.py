#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   evaluate.py
#        \author   chenghuige  
#          \date   2020-09-28 18:12:11.239056
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import heapq as hq
import random
from collections import defaultdict
import glob

import numpy as np
import sklearn
import PIL.Image as Image

import multiprocessing
import sklearn 
from itertools import count
import cv2

import wandb

import gezi
from gezi import logging, tqdm, plot
import  melt
import melt as mt
from melt.distributed import tonumpy
from gezi.metrics.image.semantic_seg import Evaluator
from husky import ProgressBar

from .config import *
from .util import *

# deprecated just use model.evaluate
def fast_eval(y, y_, x=None, step=None):
  pred = to_pred(y_)

  key_metric = 'FWIoU'
  evaluator = Evaluator(FLAGS.NUM_CLASSES)
  logger = gezi.get_summary_writer()

  res = evaluator.eval(y, pred)

  confusion_matrix = plot.confusion_matrix(evaluator.confusion_matrix, info='{}:{:.4f}'.format(key_metric, res[key_metric]))
  logger.image('0ConfusionMatrix', confusion_matrix, step)

  res = gezi.dict_prefix(res, 'Metrics/')
  return res

def eval_image(y, mark='pred'):
  CLASSES = FLAGS.CLASSES
  NUM_CLASSES = FLAGS.NUM_CLASSES
  # TODO tpu bincount带来的问题 
  if gezi.get('tpu') or not FLAGS.eval_class_per_image:
    return {}, {}, {}
  
  pixels = FLAGS.ori_image_size[0] * FLAGS.ori_image_size[1]
  bins_y = tf.math.bincount(tf.reshape(y, (-1, pixels)), minlength=NUM_CLASSES, maxlength=NUM_CLASSES, axis=-1)
  bins_y = tf.reshape(bins_y, (-1, NUM_CLASSES))
  binary_y = tf.cast(bins_y > 0, tf.int32)
  classes = tf.reduce_sum(binary_y, axis=-1)

  # 按batch 输出 方便多卡 loss_fn类似也需要如果自己loop
  info = {
    mark: classes
  }

  binary_ys = tf.unstack(binary_y, NUM_CLASSES, -1)
  pixel_ys = tf.unstack(bins_y, NUM_CLASSES, -1)
  for i in range(len(CLASSES)):
    info[f'{mark}/{CLASSES[i]}'] = binary_ys[i]
    info[f'{mark}/{CLASSES[i]}'] = pixel_ys[i] / pixels

  bins_y = tf.cast(bins_y, tf.float32)
  bins_y /= tf.cast(pixels, tf.float32)
  return info, bins_y, binary_y

def eval_images(y, y_, x):
  # TODO tpu not work..  strange, gpu单卡多卡都ok...
  #  File "/content/pikachu/projects/ai/naic2020_seg/src/evaluate.py", line 168, in eval
  #   y, y_, x['image'], x['id'] = tonumpy(y, y_, x['image'], x['id'])
  # File "/content/pikachu/utils/melt/distributed/util.py", line 135, in tonumpy
  #   x = tf.concat(x.values, axis=0).numpy()
  # ValueError: TPU function return values must all either be Operations or convertible to Tensors. Got 'None values not supported.'
  if gezi.get('tpu') or not FLAGS.eval_class_per_image:
    return {}, {}

  info_true, bins_true, binary_true = eval_image(y, mark='true')
  info_pred, bins_pred, binary_pred= eval_image(tf.argmax(y_, axis=-1), mark='pred')

  info = {}
  info.update(info_true)
  info.update(info_pred)

  ratio_classes = info['pred'] / info['true']
  info['ratio'] = ratio_classes

  intersection = tf.reduce_sum(binary_true * binary_pred, axis=-1)
  union = tf.reduce_sum(binary_true, axis=-1) + tf.reduce_sum(binary_pred, axis=-1) - intersection
  info['MIoU'] = tf.reduce_sum(intersection / union)

  others = [bins_true, binary_true, bins_pred, binary_pred]

  info = gezi.dict_prefix(info, 'IMAGE/CLASS/')
  return info, others

def eval_image_classes(binary_true, binary_pred, bins_true=None, bins_pred=None):
  CLASSES = FLAGS.CLASSES
  NUM_CLASSES = FLAGS.NUM_CLASSES

  if is_classifier():
    binary_pred = (binary_pred > FLAGS.classifier_threshold).astype(np.int32)

  metrics = {}
  total_auc, total_acc, total_recall = 0., 0., 0.
  num_classes = len(CLASSES)

  # ------- solo
  class_counts = np.sum(binary_true, axis=-1)
  solo_indices = [i for i in range(len(class_counts)) if class_counts[i] == 1]
  metrics['SOLO/ratio'] = len(solo_indices) / len(binary_true)
  solo_iou = 0
  solo_acc = 0
  solo_ious = defaultdict(float)
  solo_recalls = defaultdict(float)
  solo_counts = defaultdict(int)
  for i in solo_indices:
    solo_idx = np.argmax(binary_true[i])
    iou = binary_pred[i][solo_idx] / (np.sum(binary_true[i]) + np.sum(binary_pred[i]) - binary_pred[i][solo_idx])
    acc = binary_pred[i][solo_idx]
    solo_ious[solo_idx] += iou
    solo_recalls[solo_idx] += acc
    solo_counts[solo_idx] += 1
    solo_iou += iou
    solo_acc += acc
  solo_iou = solo_iou / len(solo_indices) if len(solo_indices) else 0.
  solo_acc = solo_acc / len(solo_indices) if len(solo_indices) else 0.
  metrics['SOLO/MIoU'] = solo_iou
  metrics['SOLO/ACC'] = solo_acc

  for i in range(NUM_CLASSES):
    if i in solo_ious:
      metrics[f'SOLO/ratio/{CLASSES[i]}'] = solo_counts[i] / len(binary_true)
      metrics[f'SOLO/MIoU/{CLASSES[i]}'] = solo_ious[i] / solo_counts[i]
      metrics[f'SOLO/Recall/{CLASSES[i]}'] = solo_recalls[i] / solo_counts[i]

  intersections = binary_true * binary_pred
  for i, class_ in enumerate(CLASSES):
    metrics[f'ACC/{CLASSES[i]}'] = np.sum(intersections[:, i]) / np.sum(binary_pred[:, i])
    metrics[f'Recall/{CLASSES[i]}'] = np.sum(intersections[:, i]) / np.sum(binary_true[:, i])
    try:
      metrics[f'AUC/{CLASSES[i]}'] = sklearn.metrics.roc_auc_score(binary_true[:, i], binary_pred[:, i])
    except Exception:
      metrics[f'AUC/{CLASSES[i]}'] = 0.
    total_auc += metrics[f'AUC/{CLASSES[i]}']
    total_acc += metrics[f'ACC/{CLASSES[i]}']
    total_recall += metrics[f'AUC/{CLASSES[i]}']
  
  metrics['AUC'] = total_auc / NUM_CLASSES
  metrics['ACC'] = total_acc / NUM_CLASSES
  metrics['Recall'] = total_recall / NUM_CLASSES

  intersection = np.sum(intersections, axis=-1)
  union = np.sum(binary_true, axis=-1) + np.sum(binary_pred, axis=-1) - intersection
  metrics['MIoU'] = np.mean(intersection / union)
  metrics = gezi.dict_prefix(metrics, 'IMAGE/CLASS/')
  if is_classifier():
    metrics = gezi.dict_prefix(metrics, 'Infos/')
    keys = ['IMAGE/CLASS/AUC', 'IMAGE/CLASS/ACC', 'IMAGE/CLASS/Recall', 'IMAGE/CLASS/MIoU']
    for key in keys:
      metrics['Metrics/' + key] = metrics['Infos/' + key]
  return metrics

def _infer(model, step=0, is_last=False):
  imgs = []
  img_names = []
  img_paths = sorted(glob.glob('../input/case/*.tif'))
  for img_path in img_paths:
    if not FLAGS.use_nir:
      img = gezi.read_tiff(img_path)[0].astype(np.float32)
    else:
      img = gezi.read_tiff(img_path, split=False).astype(np.float32)
    imgs.append(img)
    img_names.append(os.path.basename(img_path)[:-4])

  @tf.function
  def _infer_step(x):
    y_ = model(x, training=False)
    return y_

  imgs = np.asarray(imgs)
  ## has loss problem...
  # ys = model.predict_on_batch(imgs)
  # ys = model(imgs, training=False)
  strategy = mt.distributed.get_strategy()
  ys = strategy.run(_infer_step, args=(imgs,))
  ys = tonumpy(ys)
  # ys = ys.numpy()
  for i, (img, img_name, y) in enumerate(zip(imgs, img_names, ys)):
    pred = to_pred(y)
    probs = gezi.softmax(y, axis=-1)
    prob = np.max(probs, -1)
    img_ = plot.segmentation(img[:,:,:3] / 255., pred, FLAGS.CLASSES, prob=prob, title=img_name)
    tag = f'case_{step}/{i}' if not is_last else f'case/{i}'
    wandb.log({tag: wandb.Image(img_)})
    wandb.log({f'case/{i}': wandb.Image(img_)})

def _show(imgs, tag, i, logger, step):
  metric = imgs[0]
  id = str(imgs[2])
  img = imgs[3][:,:,:3]
  text = '{} {}:[{:.4f}]'.format(tag, id, metric)
  label = imgs[4]
  pred = imgs[5]
  prob =imgs[6]
  prob_label = imgs[7]
  img_ = plot.segmentation_eval(img, label, pred, FLAGS.CLASSES, prob, prob_label, title=text)
  try:
    if not FLAGS.wandb or FLAGS.wandb_tb:
      logger.image(f'{tag}/{i}', gezi.plot.tobytes(img_), step, title=text)
    else:
      wandb.log({f'{tag}/{i}': wandb.Image(img_, caption=text)})
  except Exception as e:
    # TODO 不知道为何tpu环境 如果opt_ema custom eval最后会wandb.log img出错 'The wandb backend process has shutdown'
    logging.warning(e)

    ## wandb自带交互式语义分割展示 不过似乎用处不大
    # class_labels = dict(zip(range(len(FLAGS.CLASSES)), FLAGS.CLASSES))
    # wandb.log(
    #   {
    #     f'{tag}_WANDB/{i}':
    #     wandb.Image(
    #       img, 
    #       masks={
    #         "predictions" : {
    #             "mask_data" : pred,
    #             "class_labels" : class_labels,
    #         },
    #         "ground_truth" : {
    #             "mask_data" : label,
    #             "class_labels" : class_labels,
    #         }
    #       }
    #     )
    #   }
    # )


def _tb_image(random_imgs, worst_imgs, best_imgs, step):
  logger = mt.get_summary_writer()
  
  for i in tqdm(range(len(random_imgs)), ascii=True, desc='write_random_imgs'):
    _show(random_imgs[i], 'RandomImages', i, logger, step)
  
  for i in tqdm(range(len(worst_imgs)), ascii=True, desc='write_worst_imgs'):
    _show(worst_imgs[i], 'WorstImages', i, logger, step)
  
  for i in tqdm(range(len(best_imgs)), ascii=True, desc='write_best_imgs'):
    _show(best_imgs[i], 'BestImages', i, logger, step)

class ImageEvaluator():
  def __init__(self, num_examples, display_results=True):
    self.evaluator = Evaluator(FLAGS.CLASSES)
    self.best_imgs = []
    self.worst_imgs = []

    random_img_indexes = np.asarray(range(num_examples))
    np.random.seed(FLAGS.img_show_seed)
    np.random.shuffle(random_img_indexes)
    self.random_img_indexes = random_img_indexes[:FLAGS.show_rand_imgs]

    self.step = 0
    self.tiebreaker = count()

    self.metric_image = 0
    self.metric_now = 0
    self.display_results = display_results
    self.logger = mt.get_summary_writer()

  def __call__(self, image_id, image, label, y_pred):
    key_metric = 'FWIoU'
    label = label.astype(np.int32)
    pred = to_pred(y_pred)

    metric_image = self.evaluator.eval_each(np.expand_dims(label, 0), np.expand_dims(pred, 0), metric=key_metric)[0]
    metric_now = self.evaluator.eval_once()
    self.metric_image = metric_image
    self.metric_now = metric_now
    self.step += 1
 
    if self.display_results:
      logit = y_pred
      item = [metric_image, next(self.tiebreaker), image_id, image, label, logit, pred]
      item2 = [-metric_image, next(self.tiebreaker), *item[2:]] 
      hq.heappush(self.best_imgs, item)
      hq.heappush(self.worst_imgs, item2)
      if len(self.best_imgs) > FLAGS.show_imgs:
        hq.heappop(self.best_imgs)
      if len(self.worst_imgs) > FLAGS.show_imgs:
        hq.heappop(self.worst_imgs)
      
      if self.step in self.random_img_indexes:
        _show(item, 'RandomImages', list(self.random_img_indexes).index(self.step), self.logger, self.step)

    return metric_now

  def eval_once(self):
    return self.metric_now

  def eval_each(self):
    return self.metric_image

  def finalize(self):
    if self.display_results:
      p = multiprocessing.Process(target=_tb_image, args=(self.worst_imgs, self.best_imgs, self.step))
      p.start()

# 这个只能tf了 
def eval(dataset, model, eval_step, steps, step, is_last, num_examples, loss_fn, outdir):
  CLASSES = FLAGS.CLASSES
  NUM_CLASSES = FLAGS.NUM_CLASSES

  outdir = FLAGS.model_dir
  logger = gezi.get_summary_writer()

  _infer(model, eval_step, is_last)

  key_metric = 'MIoU'
  res = {}
  infos = {}
  class_infos = {}
  #注意这里dataset可以是原始的 应该是内部会判断并做distributed转换 strategy.experimental_distribute_dataset(dataset) 从total_cm打印可以看出是distributed
  #这里输入是转换好的distribute_dataset evaluate效果是一样的
  # 自己循环需要手动加, model.evaluate实际后面的tb_image完整过程能覆盖 只是可能loss有点问题 多gpu麻烦一些 需要reduce sum 不过目前还是is_last或者。。 两个步骤都跑
  # 考虑多gpu情况model.evluate速度更加快速 注意tpu环境 drop_remainder才能跑 因为UpSampling2D 不支持dynamic 所以valid数据 可能最后一个batch多一点有一点点误差
  # 但是如果不是model.evaluate 手工loop似乎因为UpSampling2D并没有dyanmic报错..
  # 为什么tpu上面evaluate速度远慢于gpu 似乎单卡在跑 好像tpu就是只是train快
  custom_evaluate = FLAGS.custom_evaluate and (eval_step % FLAGS.custom_eval_interval == 0 or is_last) 

  if FLAGS.model_evaluate or not custom_evaluate:
    res = model.evaluate(dataset, steps=FLAGS.model_evaluate_count or steps, return_dict=True, callbacks=[ProgressBar(f'evaluate_{eval_step}')], verbose=0)
    res = gezi.dict_rename(res, 'loss', 'Loss')

    cm = mt.distributed.sum_merge(gezi.get('info')['metrics'][0].get_cm())

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
      logging.debug('{:20s}'.format(CLASSES[i]), '|true:{:.4f} pred:{:.4f} acc:{:.4f} recall:{:.4f} iou:{:.4f} fwiou:{:.4f}|'.format(classes_ratio_true[i], classes_ratio_pred[i], acc, recall, iou, fwiou))
      # 因为后面infos统一 / num_examples  HACK
      infos[f'IoU/{CLASSES[i]}'] = iou
      infos[f'FWIoU/{CLASSES[i]}'] = fwiou
      infos[f'ACC/{CLASSES[i]}'] = acc
      infos[f'REC/{CLASSES[i]}'] = recall
    gezi.dict_mul(infos, num_examples)
    logging.debug('model_evaluate:', res)
  
    # # 展现 confusion matrix
    cm_args = dict(
          classes=CLASSES,
          normalize='true', 
          info='{}:{:.4f}'.format(key_metric, res[key_metric]),
          title='',
          #show=FLAGS.show_imgs_online,
          img_size=15,
    )

    # notbook训练过程可以不展示 最后两次单独fit 再展示
    if FLAGS.write_confusion_matrix and is_last:    
      cm_args['title'] = 'Recall'
      confusion_matrix = plot.confusion_matrix(cm, **cm_args)
      if not FLAGS.wandb:
        confusion_matrix = gezi.plot.tobytes(confusion_matrix)
        logger.image('ConfusionMatrix/Recall', confusion_matrix, eval_step)
      else:
        wandb.log({'ConfusionMatrix/Recall': wandb.Image(confusion_matrix)})
      cm_args['title'] = 'Precision'
      cm_args['normalize'] = 'pred'
      confusion_matrix = plot.confusion_matrix(cm, **cm_args)
      if not FLAGS.wandb:
        confusion_matrix = gezi.plot.tobytes(confusion_matrix)
        logger.image('ConfusionMatrix/Precesion', confusion_matrix, eval_step)
      else:
        wandb.log({'ConfusionMatrix/Precision': wandb.Image(confusion_matrix)})
      cm_args['title'] = 'All'
      cm_args['normalize'] = 'all'
      confusion_matrix = plot.confusion_matrix(cm / num_examples, **cm_args)
      if not FLAGS.wandb:
        confusion_matrix = gezi.plot.tobytes(confusion_matrix)
        logger.image('ConfusionMatrix/All', confusion_matrix, eval_step)
      else:
        wandb.log({'ConfusionMatrix/All': wandb.Image(confusion_matrix)})

  if custom_evaluate:
    tb_image = FLAGS.tb_image and (eval_step % FLAGS.tb_image_interval == 0 or is_last) 
    evaluator = Evaluator(NUM_CLASSES)
    # evaluator = Evaluator(CLASSES)
    best_imgs = []
    worst_imgs = []
    random_imgs = []
    metrics = []

    # fixed random seed for compare ? TODO
    random_img_indexes = np.asarray(range(num_examples))
    np.random.seed(FLAGS.img_show_seed)
    np.random.shuffle(random_img_indexes)
    random_img_indexes = random_img_indexes[:FLAGS.show_rand_imgs]

    # tpu 单独运行测试开始报错 不知道是不是引入返回info带来的 gpu测试单卡多卡都ok
    # File "/content/pikachu/utils/melt/distributed/util.py", line 133, in tonumpy
    #   x[key] = tf.concat(x[key].values, axis=0).numpy()
    @tf.function
    def _eval_step(x, y):
      y_ = model(x, training=False)
      info, others = eval_images(y, y_, x)
      return y_, info, others

    strategy = mt.distributed.get_strategy()
    dataset = strategy.experimental_distribute_dataset(dataset)
    cur_step = 0
    
    bins_trues, binary_trues, bins_preds, binary_preds = [], [], [], []

    tiebreaker = count()
    t = tqdm(enumerate(dataset), total=steps, ascii=True, desc= 'eval_loop')
    examples = 0
    for step_, (x, y) in t:
      if step_ == steps:
        break
      examples += mt.eval_batch_size()
        
      # y_ = model(x)
      # y_ = eval_step(x, y)
      y_, info, others = strategy.run(_eval_step, args=(x, y))

      y, y_, x['image'], x['id'] = tonumpy(y, y_, x['image'], x['id'])

      if examples > num_examples:
        diff = num_examples - examples
        examples = num_examples
        y = y[:diff]
        y_ = y_[:diff]
        keys = ['image', 'id']
        for key in keys:
          x[key] = x[key][:diff]

      x['image'] = x['image'].astype(np.float32) / 255.

      info_ = {}
      if info:
        info = tonumpy(info)
        info = gezi.dict_sum(info)
        gezi.dict_add(infos, info)

        info_ = gezi.dict_div_copy(infos, examples)
        keys = ['IMAGE/CLASS/ratio', 'IMAGE/CLASS/MIoU']
        info_ = dict([(x, info_[x]) for x in info_ if x in keys])

      if others:
        bins_true, binary_true, bins_pred, binary_pred = tonumpy(*others)
        bins_trues.append(bins_true)
        binary_trues.append(binary_true)
        bins_preds.append(bins_pred)
        binary_preds.append(binary_pred)

      x['id'] = gezi.squeeze(x['id'])
      # (x, x, 1) -> (x, x)
      y = gezi.squeeze(y)

      if FLAGS.write_inter_results:
        write_inter_results(x['id'], y_, outdir, mode='valid', masks=y)

      pred = to_pred(y_)
      probs = gezi.softmax(y_, -1)
      prob = probs.max(-1)
      # 注意这里预先计算好 不使用gezi.plot.segmentation_eval_logits接口避免内存OOM
      prob_label = gezi.lookup_nd(probs, y.astype(np.int32))

      if FLAGS.write_valid_results:
        write_results(x['id'], y_, outdir, mode='valid')

      metric = evaluator.eval_each(y, pred, metric=key_metric)
      metrics += [metric]

      t.set_postfix({key_metric: evaluator.eval_once(key_metric), **info_})

     
      if tb_image:
        for i in range(len(y)):
          # try:
          # https://stackoverflow.com/questions/39504333/python-heapq-heappush-the-truth-value-of-an-array-with-more-than-one-element-is
          item = [metric[i], next(tiebreaker), x['id'][i], x['image'][i], y[i].astype(np.int32), pred[i], prob[i], prob_label[i]]
          item2 = [-metric[i], next(tiebreaker), *item[2:]] 
          hq.heappush(best_imgs, item)
          hq.heappush(worst_imgs, item2)
          if len(best_imgs) > FLAGS.show_best_imgs:
            hq.heappop(best_imgs)
          if len(worst_imgs) > FLAGS.show_worst_imgs:
            hq.heappop(worst_imgs)
          
          if cur_step in random_img_indexes:
            _show(item, 'RandomImages', list(random_img_indexes).index(cur_step), logger, eval_step)
            # random_imgs.append(item)
          # except Exception as e:
          #   logging.error(e)

          cur_step += 1

    if others:
      binary_true, binary_pred = np.concatenate(binary_trues, axis=0), np.concatenate(binary_preds, axis=0)
      bins_true, bins_pred = np.concatenate(bins_trues, axis=0), np.concatenate(bins_preds, axis=0)
      class_infos = eval_image_classes(binary_true, binary_pred, bins_true, bins_pred)
    
    for item in worst_imgs:
      item[0] = -item[0]
        
    res.update(evaluator.eval_once())
    res[key_metric + '2'] = np.mean(np.concatenate(metrics))
    
    if tb_image:
      if not is_last:
        p = multiprocessing.Process(target=_tb_image, args=(random_imgs, worst_imgs, best_imgs, eval_step))
        p.start()
      else:
        _tb_image(random_imgs, worst_imgs, best_imgs, eval_step)

  infos = gezi.dict_div(infos, num_examples)
  infos.update(class_infos)
  infos = gezi.dict_prefix(infos, 'Infos/')
  gezi.set('Metrics', res)
  res = gezi.dict_prefix(res, 'Metrics/')
  res.update(infos)

  writer = gezi.get('tfrec_valid_writer')
  if writer:
    writer.close()

  return res
  
def get_eval_fn():
  if is_classifier():
    return eval_image_classes

  if FLAGS.fast_eval:
    # dereciated just use eval
    return fast_eval
  else:
    return eval
