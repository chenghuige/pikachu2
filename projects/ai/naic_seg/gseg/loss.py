#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   loss.py
#        \author   chenghuige  
#          \date   2020-09-28 18:11:04.187754
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np
import melt as mt
from gezi import logging
import gseg
from .config import *
from .util import *

def get_loss_fn():
  if is_classifier():
    def _loss_fn(y_true, y_pred):
      losses = tf.keras.losses.BinaryCrossentropy(from_logits=FLAGS.from_logits, label_smoothing=FLAGS.label_smoothing, reduction=tf.keras.losses.Reduction.NONE)(y_true, y_pred)
      return mt.reduce_over(losses)
    return _loss_fn

  # sigmoid not good result, depreated
  if FLAGS.class_lookup and FLAGS.loss_fn == 'sigmoid':
    class_lookup = tf.constant(CLASS_LOOKUP, dtype=tf.int32)
    def _loss_fn(y_true, y_pred):
      y_true = tf.nn.embedding_lookup(class_lookup, tf.squeeze(y_true, -1))
      x = tf.keras.losses.BinaryCrossentropy(from_logits=FLAGS.from_logits, label_smoothing=FLAGS.label_smoothing, reduction=tf.keras.losses.Reduction.NONE)(y_true, y_pred)
      x = tf.reduce_mean(tf.reshape(x, (-1, x.shape[1] * x.shape[2])), -1)
      x = mt.reduce_over(x)
      return x
    return _loss_fn

  # TODO FIXME 为何在外部用input不行..
  # loss_fn = _get_loss_fn(input, model)
  loss_fn = _get_loss_fn()
  distill_loss_fn = keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)

  def _loss_fn(y_true, y_pred, input=None, model=None):
    mask = None
    weights = None
    if FLAGS.mix_dataset:
      ## 20201201 目前已经在tfrecord制作阶段完成了转换
      # if FLAGS.class_lookup_soft:
      #   class_lookup = tf.constant(CLASS_LOOKUP_SOFT, dtype=tf.float32)
      # else:
      #   class_lookup = tf.constant(CLASS_LOOKUP, dtype=tf.int32)

      mark_v2 = tf.cast(tf.equal(input['src'], 2), tf.float32)[:, tf.newaxis, tf.newaxis, :]
      mask = mark_v2 * tf.nn.embedding_lookup(tf.ones_like(LABLE_MASK, tf.float32), tf.cast(tf.squeeze(y_true, -1), tf.int32)) \
            + (1 - mark_v2) * tf.nn.embedding_lookup(tf.cast(LABLE_MASK, tf.float32), tf.cast(tf.squeeze(y_true, -1), tf.int32))
      # def onehot(x):
      #   x = tf.cast(tf.squeeze(x, -1), tf.int32)
      #   mark_v2 = tf.cast(tf.equal(input['src'], 2), tf.float32)[:, tf.newaxis, tf.newaxis, :]
      #   x = tf.cast(tf.one_hot(x, y_pred.shape[-1]), tf.float32) * mark_v2 + tf.cast(tf.nn.embedding_lookup(class_lookup, x), tf.float32) * (1 - mark_v2)
      #   return x
      # y_true = onehot(y_true)

    if FLAGS.dataset_weights:
      # weights = tf.ones_like(input['src'], dtype=tf.float32)
      # TODO more then 2 datasets ?
      index = tf.squeeze(input['src'], -1)
      mark_v2 = tf.cast(tf.equal(index, 2), tf.float32)
      weights = FLAGS.dataset_weights[0] * mark_v2 + FLAGS.dataset_weights[1] * (1 - mark_v2)
      weights = weights[:, tf.newaxis, tf.newaxis]

    if FLAGS.weights_strategy:
      if FLAGS.weights_strategy == 1:
        weights = tf.reduce_sum(tf.cast(input['bins'] > 0, tf.float32), axis=-1) / 4.
      elif FLAGS.weights_strategy == 2:
        weights = tf.squeeze(tf.math.log(tf.cast(input['components'] + 1, tf.float32)), -1)
      else:
        raise ValueError(FLAGS.weights_strategy)

    if mask is not None:
      loss_fn_ = _get_loss_fn(mask)
      x = loss_fn_(y_true, y_pred)
    else:
      x = loss_fn(y_true, y_pred)
    
    if FLAGS.distill and K.learning_phase():
      # https://github.com/PaddlePaddle/PaddleClas/blob/master/docs/zh_CN/advanced_tutorials/distillation/distillation.md
      # with tf.device('/cpu:0'):
      # TODO teacher(input['image'], training=FLAGS.teacher_train_mode) tf2.3可以2.4报错 Tensor.op is meaningless when eager execution is enabled. 
      if FLAGS.teacher:
        teacher = get_teacher()
        # teacher = model.teacher
        if FLAGS.teacher_splits == 1:
          y_true_ = teacher(input['image'], training=FLAGS.teacher_train_mode)
        else:
          inputs = tf.split(input['image'], FLAGS.teacher_splits, axis=0)
          res = [teacher(input_, training=FLAGS.teacher_train_mode) for input_ in inputs]
          y_true_ = tf.concat(res, axis=0)
      else:
        y_true_ = input['pred']
      
      y_logit = y_true_
      y_true_ = tf.nn.softmax(y_true_ / FLAGS.temperature)
      y_pred_ = tf.nn.softmax(y_pred / FLAGS.temperature)
      distill_loss = distill_loss_fn(y_true_, y_pred_)

      if FLAGS.teacher_thre:
        y_hard = tf.reduce_max(tf.nn.softmax(y_logit), axis=-1)
        teacher_mask = tf.cast(y_hard > FLAGS.teacher_thre, tf.float32)
        distill_loss *= teacher_mask

      x = (1- FLAGS.teacher_rate) * x + FLAGS.teacher_rate * distill_loss
    
    if weights is not None and len(weights.shape) == len(x.shape):
      x *= weights

    if len(x.shape) == 3:
      x = tf.reduce_mean(x, [1, 2])

    if weights is not None and len(weights.shape) == len(x.shape):
      x *= weights

    if len(x.shape) == 1:
      x = mt.reduce_over(x)

    if FLAGS.ce_loss_rate:
      # y_true, y_pred = _preprocess(y_true, y_pred, from_logits=FLAGS.from_logits, label_smoothing=FLAGS.label_smoothing)
      # ce_loss = mt.reduce_over(tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0.))
      def _ce_loss_fn(y_true, y_pred):
        y_true = _to_onehot_label(y_true, y_pred)
        return tf.keras.losses.CategoricalCrossentropy(from_logits=FLAGS.from_logits, label_smoothing=FLAGS.label_smoothing, reduction=tf.keras.losses.Reduction.NONE)(y_true, y_pred)
      ce_loss = _ce_loss_fn(y_true, y_pred)
      if FLAGS.ce_loss_rate < 1:
        x = x * (1. - FLAGS.ce_loss_rate) + ce_loss * FLAGS.ce_loss_rate
      elif FLAGS.ce_loss_rate > 1:
        x = x * (FLAGS.ce_loss_rate - 1.) + ce_loss
      else:
        x += ce_loss
    if FLAGS.bce_loss_rate:
      pred = tf.keras.layers.GlobalAveragePooling2D()(y_pred)
      pred = tf.reshape(pred, (-1, FLAGS.NUM_CLASSES))
      y = pixel2class_label(y_true, input)
      bce_loss = mt.reduce_over(keras.losses.binary_crossentropy(y, pred, from_logits=FLAGS.from_logits))
      x += bce_loss * FLAGS.bce_loss_rate
    if FLAGS.multi_rate:
      y = pixel2class_label(y_true, input)
      pred = model.y2
      if FLAGS.use_class_weights:
        losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(y, tf.float32), logits=pred)
        class_weights = gezi.get('class_weights')
        losses *= class_weights
      else:
        if y.shape[-1] != pred.shape[-1]:
          y = y[:, :pred.shape[-1]]
        losses = tf.keras.losses.BinaryCrossentropy(from_logits=FLAGS.from_logits, label_smoothing=FLAGS.label_smoothing, reduction=tf.keras.losses.Reduction.NONE)(y, pred)
      bce_loss = mt.reduce_over(losses)
      x += bce_loss * FLAGS.multi_rate
    if FLAGS.components_loss_rate:
      import tensorflow_addons as tfa
      num_pixels = FLAGS.image_size[0] * FLAGS.image_size[1]
      components_pred = tf.reduce_max(tf.reshape(tfa.image.connected_components(tf.argmax(y_pred, axis=-1)), (-1, num_pixels)), axis=-1)
      components_true = input['components']
      components_pred = tf.cast(tf.math.minimum(components_pred, 10), tf.float32) / 10.
      components_true = tf.cast(tf.math.minimum(components_true, 10), tf.float32) / 10.
      components_loss = tf.losses.mean_squared_error(components_true, components_pred)
      x += components_loss * FLAGS.components_loss_rate
    return x

  return _loss_fn

# def _get_loss_fn(input=None, model=None):
def _get_loss_fn(mask=None):
  loss_name = FLAGS.loss_fn
  # logging.info('from_logits', FLAGS.from_logits)
  if not loss_name or loss_name in ['default', 'ce']:
    if mask is None:
      # 所有的都按照转换onehot的格式作为y_true输入 loss计算
      def _loss_fn(y_true, y_pred):
        y_true = _to_onehot_label(y_true, y_pred)
        return tf.keras.losses.CategoricalCrossentropy(from_logits=FLAGS.from_logits, label_smoothing=FLAGS.label_smoothing, reduction=tf.keras.losses.Reduction.NONE)(y_true, y_pred)
      return _loss_fn
      # else:
      #   # return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=FLAGS.from_logits)
      #   return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=FLAGS.from_logits, reduction=tf.keras.losses.Reduction.NONE)
      #   # return lambda y_true, y_pred: keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=FLAGS.from_logits)
    else:
      return lambda y_true, y_pred: weighted_categorical_crossentropy_loss(y_true, y_pred, mask, from_logits=FLAGS.from_logits, label_smoothing=FLAGS.label_smoothing)

  elif loss_name in ['weighted', 'weighted_ce']:
    # class_weights = [1.] * FLAGS.NUM_CLASSES
    # class_weights[2], class_weights[3], class_weights[-1] = 1.1, 1.2, 1.3
    class_weights = gezi.get('class_weights')
    if mask is not None:
      class_weights = mask * class_weights
    # TODO 再试一下 完全按照 class分布weight的loss
    return lambda y_true, y_pred: weighted_categorical_crossentropy_loss(y_true, y_pred, class_weights, from_logits=FLAGS.from_logits, label_smoothing=FLAGS.label_smoothing)
  
  elif loss_name == 'tanimoto':
    class_weights = gezi.get('class_weights')
    if mask is not None:
      class_weights = mask * class_weights
    return lambda y_true, y_pred: tanimoto_loss(y_true, y_pred, class_weights, from_logits=FLAGS.from_logits, label_smoothing=FLAGS.label_smoothing)

  elif loss_name == 'weighted_dice':
    class_weights = gezi.get('class_weights')
    if mask is not None:
      class_weights = mask * class_weights
    return lambda y_true, y_pred: weighted_dice_loss(y_true, y_pred, class_weights, from_logits=FLAGS.from_logits, label_smoothing=FLAGS.label_smoothing)

  elif loss_name == 'sigmoid': # sigmoid much worse result then softmax
    def _loss_fn(y_true, y_pred):
      # [256,256,1] -> [256,256]
      y_true = tf.squeeze(y_true, -1)
      # [256,256,15]
      y_true = tf.one_hot(tf.cast(y_true, tf.int32), y_pred.shape[-1])
      x = tf.keras.losses.BinaryCrossentropy(from_logits=FLAGS.from_logits, label_smoothing=FLAGS.label_smoothing, reduction=tf.keras.losses.Reduction.NONE)(y_true, y_pred)
      return x
    return _loss_fn

  if loss_name == 'focal':
    import focal_loss
    return focal_loss.SparseCategoricalFocalLoss(gamma=2, from_logits=FLAGS.from_logits, reduction=tf.keras.losses.Reduction.NONE)
     
  return lambda y_true, y_pred: _other_loss_fn(y_true, y_pred, loss_name)

def _other_loss_fn(y_true, y_pred, loss_name):
  if len(y_true.shape) == len(y_pred.shape):
    y_true = tf.squeeze(y_true, -1)

  # if FLAGS.from_logits:
  #   y_pred = tf.nn.softmax(y_pred, axis=-1)

  # y_true = tf.cast(y_true, y_pred.dtype)

  if hasattr(gseg.loss, loss_name + '_loss'):
    loss_fn = getattr(gseg.loss, loss_name + '_loss')
    return loss_fn(y_true, y_pred)
  else:
    raise ValueError(loss_name)  

def _to_onehot_label(y_true, y_pred):
  # class_lookup only for data version 1 (8 classes) to adapt for data version 2 (15 classes)
  if FLAGS.class_lookup:
    if FLAGS.class_lookup_soft:
      class_lookup = tf.constant(CLASS_LOOKUP_SOFT, dtype=tf.float32)
    else:
      class_lookup = tf.constant(CLASS_LOOKUP, dtype=tf.int32)

  if y_true.shape[1:] != y_pred.shape[1:]:
    if len(y_true.shape) == len(y_pred.shape):
      y_true = tf.squeeze(y_true, -1)
    
    if FLAGS.class_lookup:
      y_true = tf.nn.embedding_lookup(class_lookup, y_true)
    else:
      y_true = tf.one_hot(tf.cast(y_true, tf.int32), y_pred.shape[-1])
  return y_true

def _preprocess(y_true, y_pred, from_logits=True, label_smoothing=0.):
  y_true = _to_onehot_label(y_true, y_pred)

  if label_smoothing:
    smooth_val = tf.ones_like(y_true, tf.float32) * (label_smoothing / (y_pred.shape[-1] - 1))
    mask = tf.cast(tf.equal(y_true, 1), tf.float32)
    y_true = mask * (1 - label_smoothing) + (1 - mask) * smooth_val
    # print(y_true.shape)
    # print(y_true[0][0][0])
  else:
    y_true = tf.cast(y_true, tf.float32)

  if from_logits:
    y_pred = tf.nn.softmax(y_pred, axis=-1)

  y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())  # To avoid unwanted behaviour in K.log(y_pred)

  return y_true, y_pred

# https://cloud.tencent.com/developer/article/1652398
# TODO 这里默认不reduce REDUCTION=None模式
def weighted_categorical_crossentropy_loss(y_true, y_pred, class_weights, from_logits=True, label_smoothing=0.):
  """
  weighted_categorical_crossentropy between an output and a target
  loss=-weight*y*log(y')
  :param Y_pred:A tensor resulting from a softmax
  :param Y_gt:A tensor of the same shape as `output`
  :param weights:numpy array of shape (C,) where C is the number of classes
  :return:categorical_crossentropy loss
  Usage:
  weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
  """
  y_true, y_pred = _preprocess(y_true, y_pred, from_logits=from_logits, label_smoothing=label_smoothing)    

  if not isinstance(class_weights, tf.Tensor):
    class_weights = tf.constant(np.asarray(class_weights), dtype=tf.float32)

  loss = - y_true * tf.math.log(y_pred) * class_weights
  loss = tf.reduce_sum(loss, axis=-1)
  return loss

def tanimoto_loss(y_true, y_pred, class_weights, from_logits=True, label_smoothing=0.):
  """
  Weighted Tanimoto loss.
  Defined in the paper "ResUNet-a: a deep learning framework for semantic segmentation of remotely sensed data",
  under 3.2.4. Generalization to multiclass imbalanced problems. See https://arxiv.org/pdf/1904.00592.pdf
  Used as loss function for multi-class image segmentation with one-hot encoded masks.
  :param class_weights: Class weight coefficients (Union[list, np.ndarray, tf.Tensor], len=<N_CLASSES>)
  :return: Weighted Tanimoto loss function (Callable[[tf.Tensor, tf.Tensor], tf.Tensor])
  """
  y_true, y_pred = _preprocess(y_true, y_pred, from_logits=from_logits, label_smoothing=label_smoothing)    

  if not isinstance(class_weights, tf.Tensor):
    class_weights = tf.constant(np.asarray(class_weights), dtype=tf.float32)

  numerator = y_true * y_pred * class_weights
  numerator = K.sum(numerator, axis=-1)

  denominator = (y_true**2 + y_pred**2 - y_true * y_pred) * class_weights
  denominator = K.sum(denominator, axis=-1)
  return 1 - numerator / denominator

def weighted_dice_loss(y_true, y_pred, class_weights, from_logits=True, label_smoothing=0.):
  y_true, y_pred = _preprocess(y_true, y_pred, from_logits=from_logits, label_smoothing=label_smoothing)    

  if not isinstance(class_weights, tf.Tensor):
    class_weights = tf.constant(np.asarray(class_weights), dtype=tf.float32) 

  numerator = y_true * y_pred * class_weights  # Broadcasting
  numerator = 2. * K.sum(numerator, axis=-1)

  denominator = (y_true + y_pred) * class_weights # Broadcasting
  denominator = K.sum(denominator, axis=-1)

  return 1 - numerator / denominator

def lovasz_softmax_loss(y_true, y_pred):
  return lovasz_softmax(y_pred, y_true)

def lovasz_softmax_per_image_loss(y_true, y_pred):
  return lovasz_softmax(y_pred, y_true, per_image=True)
    
# https://github.com/keras-team/keras/issues/9395
# Hey guys, I found a way to implement multi-class dice loss, I get satisfying segmentations now. I implemented the loss as explained in ref : 
# this paper describes the Tversky loss, a generalised form of dice loss, which is identical to dice loss when alpha=beta=0.5
# Here is my implementation, for 3D images:
# Ref: salehi17, "Twersky loss function for image segmentation using 3D FCDN"
# -> the score is computed for each class separately and then summed
# alpha=beta=0.5 : dice coefficient
# alpha=beta=1   : tanimoto coefficient (also known as jaccard)
# alpha+beta=1   : produces set of F*-scores
# implemented by E. Moebel, 06/04/18

# def tversky_loss(y_true, y_pred):
#   if y_true.shape[1:] != y_pred.shape[1:]:
#     y_true = tf.one_hot(y_true, y_pred.shape[-1])

#   alpha = 0.5
#   beta  = 0.5

#   ones = K.ones_like(y_true)
#   p0 = y_pred      # proba that voxels are class i
#   p1 = ones-y_pred # proba that voxels are not class i
#   g0 = y_true
#   g1 = ones-y_true

#   num = K.sum(p0*g0, (0,1,2))
#   den = num + alpha*K.sum(p0*g1,(0,1,2)) + beta*K.sum(p1*g0,(0,1,2))

#   T = K.sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]

#   Ncl = K.cast(K.shape(y_true)[-1], 'float32')
#   return Ncl - T

def dice_coeff(y_true, y_pred):
  if y_true.shape[1:] != y_pred.shape[1:]:
    y_true = tf.one_hot(y_true, y_pred.shape[-1])

  Ncl = y_pred.shape[-1]
  w = tf.zeros((Ncl,))
  w = K.sum(y_true, axis=(0,1,2))
  w = 1/(w**2+0.000001)
  # Compute gen dice coef:
  numerator = y_true*y_pred
  numerator = w*K.sum(numerator,(0,1,2,3))
  numerator = K.sum(numerator)

  denominator = y_true+y_pred
  denominator = w*K.sum(denominator,(0,1,2,3))
  denominator = K.sum(denominator)

  gen_dice_coef = 2*numerator/denominator

  return gen_dice_coef

# def dice_loss(y_true, y_pred):
#   return 1 - dice_coeff(y_true, y_pred)

# https://github.com/baudcode/tf-semantic-segmentation/blob/master/tf_semantic_segmentation/losses/dice.py
def dice_loss(y_true, y_pred):
  """ F1 Score """
  if y_true.shape[1:] != y_pred.shape[1:]:
    y_true = tf.one_hot(y_true, y_pred.shape[-1])
  numerator = 2 * tf.reduce_sum(y_true * y_pred, -1)
  denominator = tf.reduce_sum(y_true + y_pred, -1)

  r = 1 - (numerator + 1) / (denominator + 1)
  return tf.cast(r, tf.float32)


""" Tversky index (TI) is a generalization of Dice’s coefficient. TI adds a weight to FP (false positives) and FN (false negatives). """
def tversky_loss(y_true, y_pred, beta=0.7):
  numerator = tf.reduce_sum(y_true * y_pred)
  denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)

  r = 1 - (numerator + 1) / (tf.reduce_sum(denominator) + 1)
  return tf.cast(r, tf.float32)

def focal_tversky_loss(y_true, y_pred, beta=0.7, gamma=0.75):
  loss = tversky_loss(y_true, y_pred, beta=beta)
  return tf.pow(loss, gamma)

# https://github.com/bermanmaxim/LovaszSoftmax/blob/master/tensorflow/lovasz_losses_tf.py
def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard


# --------------------------- BINARY LOSSES ---------------------------


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        def treat_image(log_lab):
            log, lab = log_lab
            log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
            log, lab = flatten_binary_scores(log, lab, ignore)
            return lovasz_hinge_flat(log, lab)
        losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
        loss = losses
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """

    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = lovasz_grad(gt_sorted)
        loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
        return loss

    # deal with the void prediction case (only void pixels)
    loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                   lambda: tf.reduce_sum(logits) * 0.,
                   compute_loss,
                   strict=True,
                   name="loss"
                   )
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = tf.reshape(scores, (-1,))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return scores, labels
    valid = tf.not_equal(labels, ignore)
    vscores = tf.boolean_mask(scores, valid, name='valid_scores')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vscores, vlabels


# --------------------------- MULTICLASS LOSSES ---------------------------


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None, order='BHWC'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, H, W, C] or [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
      order: use BHWC or BCHW
    """
    if per_image:
        def treat_image(prob_lab):
            prob, lab = prob_lab
            prob, lab = tf.expand_dims(prob, 0), tf.expand_dims(lab, 0)
            prob, lab = flatten_probas(prob, lab, ignore, order)
            return lovasz_softmax_flat(prob, lab, classes=classes)
        losses = tf.map_fn(treat_image, (probas, labels), dtype=tf.float32)
        # loss = tf.reduce_mean(losses)
        loss = losses 
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore, order), classes=classes) 
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    C = probas.shape[1]
    losses = []
    present = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = tf.cast(tf.equal(labels, c), probas.dtype)  # foreground for class c
        if classes == 'present':
            present.append(tf.reduce_sum(fg) > 0)
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = tf.abs(fg - class_pred)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort_{}".format(c))
        fg_sorted = tf.gather(fg, perm)
        grad = lovasz_grad(fg_sorted)
        losses.append(
            tf.tensordot(errors_sorted, tf.stop_gradient(grad), 1, name="loss_class_{}".format(c))
                      )
    if len(class_to_sum) == 1:  # short-circuit mean when only one class
        return losses[0]
    losses_tensor = tf.stack(losses)
    if classes == 'present':
        present = tf.stack(present)
        losses_tensor = tf.boolean_mask(losses_tensor, present)
    loss = tf.reduce_mean(losses_tensor)
    return loss


def flatten_probas(probas, labels, ignore=None, order='BHWC'):
    """
    Flattens predictions in the batch
    """
    if len(probas.shape) == 3:
        probas, order = tf.expand_dims(probas, 3), 'BHWC'
    if order == 'BCHW':
        probas = tf.transpose(probas, (0, 2, 3, 1), name="BCHW_to_BHWC")
        order = 'BHWC'
    if order != 'BHWC':
        raise NotImplementedError('Order {} unknown'.format(order))
    C = probas.shape[3]
    probas = tf.reshape(probas, (-1, C))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return probas, labels
    valid = tf.not_equal(labels, ignore)
    vprobas = tf.boolean_mask(probas, valid, name='valid_probas')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vprobas, vlabels
  
