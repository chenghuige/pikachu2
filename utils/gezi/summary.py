#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   summary.py
#        \author   chenghuige
#          \date   2019-07-22 19:09:08.471338
#   \Description
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import tensorflow as tf
import numpy as np
import scipy.misc
import PIL.Image as Image
#import cv2
import datetime
from absl import flags

FLAGS = flags.FLAGS

from io import BytesIO
import matplotlib.pyplot as plt

import gezi

# https://pytorch.org/docs/stable/tensorboard.html
# tensorboardX might be the same as from torch.utils.tensorboard import SummaryWriter


# For compat, you can just use gezi.summary.SummayWriter or gezi.SummaryWriter
def get_writer(log_dir, set_walltime=False):
  # can be used both for eager and non eager, need to wrap more for supporting image,hist
  return SummaryWriter(log_dir, is_tf=False, set_walltime=set_walltime)


#   if tf.executing_eagerly():
#     return EagerSummaryWriter(log_dir)
#   else:
#     return SummaryWriter(log_dir, is_tf=True)

## Depreciated
# summary = tf.contrib.summary
summary = tf.compat.v2.summary


class EagerSummaryWriter(object):

  def __init__(self, log_dir):
    """Create a summary writer logging to log_dir."""
    gezi.try_mkdir(log_dir)
    try:
      self.writer = tf.summary.create_file_writer(logdir=log_dir)
    except Exception:
      self.writer = tf.compat.v2.summary.create_file_writer(logdir=log_dir)

  def scalar(self, tag, value, step):
    """Log a scalar variable."""
    with self.writer.as_default():
      summary.scalar(tag, value, step=step)
    #   self.writer.flush()

  # TODO other summaries like image


class SummaryWriter(object):

  # tf 2.5.1 is_tf=True --mode=valid 最后core
  # File "/home/huigecheng/.local/lib/python3.6/site-packages/tensorboard/plugins/scalar/summary_v2.py", line 91 in scalar
  def __init__(self, log_dir, set_walltime=True, is_tf=False):
    """Create a summary writer logging to log_dir."""
    gezi.try_mkdir(log_dir)
    self.is_tf = is_tf
    self.is_eager_summary = False
    if is_tf:
      if tf.executing_eagerly():
        self.writer = EagerSummaryWriter(log_dir)
        self.is_eager_summary = True
      else:
        self.writer = tf.compat.v1.summary.FileWriter(log_dir)
    else:
      from torch.utils.tensorboard import SummaryWriter
      #from tensorboardX import SummaryWriter
      self.writer = SummaryWriter(log_dir)
    self.set_walltime = set_walltime

  def flush(self):
    # return
    if not self.is_eager_summary:
      self.writer.flush()

  def scalars(self, results, step=None, walltime=None):
    if (not step) and ('step' in results):
      step = results['step']
    if not step:
      step = 1
    for key, val in results.items():
      if key != 'step':
        try:
          self.scalar(key, val, step, walltime)
        except Exception:
          pass

  def write(self, results, step=None, walltime=None):
    self.scalars(results, step, walltime)

  def scalar(self, tag, value, step=1, walltime=None):
    # print('----------------------------', step, tag)
    if self.set_walltime:
      if walltime == 0:
        walltime = None
      elif walltime == None:
        try:
          walltime = datetime.datetime.strptime(
              str(FLAGS.valid_hour),
              '%Y%m%d%H').replace(tzinfo=datetime.timezone.utc).timestamp() if FLAGS.valid_hour else None
        except Exception:
          try:
            walltime = datetime.datetime.strptime(
              str(FLAGS.valid_day),
              '%Y%m%d').replace(tzinfo=datetime.timezone.utc).timestamp() if FLAGS.valid_day else None
          except Exception:
            walltime = None
    if self.is_tf:
      if tf.executing_eagerly():
        self.writer.scalar(tag, value, step)
      else:
        """Log a scalar variable."""
        summary = tf.compat.v1.Summary(
            value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)
    else:
      if hasattr(value, 'detach'):
        value = value.detach().cpu()
      elif tf.executing_eagerly() and hasattr(value, 'numpy'):
        value = value.numpy()
      self.writer.add_scalar(tag, value, step, walltime)
    self.flush()

  def gen_image(self, img, title=''):

    def _gen_image(img, s, title):
      ig, ax = plt.subplots()
      ax.set_axis_off()
      if title:
        ax.set_title(title, fontsize=40)
      ax.imshow(img)
      # plt.colorbar()
      plt.axis('off')
      plt.savefig(s, format='png', bbox_inches='tight')

    if not isinstance(img, BytesIO):
      s = BytesIO()
      if isinstance(img, str):
        img = Image.open(img).convert('RGB')
        # if not title:
        #     img.save(s, format='png')
        # else:
        img = np.asarray(img)
        img = img / 255.
        _gen_image(img, s, title)
      else:
        # if not title:
        #     Image.fromarray(img).save(s, format='png')
        # else:
        _gen_image(img, s, title)
    else:
      s = img
    return s

  def image(self, tag, img, step=1, title=''):
    img = self.gen_image(img, title)
    image = Image.open(img)
    img_array = np.asarray(image)
    self.writer.add_image(tag, img_array, step, dataformats='HWC')
    self.flush()

  def images(self, tag, images, step=1, title='', titles=[], concat=True):
    """Log a list of images."""
    if not concat:
      if not isinstance(images, (list, tuple)):
        images = [images]

      if titles:
        assert len(images) == len(titles)
      else:
        texts = [''] * len(images)
      for i, (img, text) in enumerate(zip(images, texts)):
        self.image(f'{tag}/{i}', img, step, text, colorbar)
    else:
      imgs = [self.gen_image(img) for img in images]
      imgs = [np.asarray(Image.open(img)) for img in imgs]
      gezi.plot.display_images(imgs, title=title, titles=titles)
      img = BytesIO()
      plt.savefig(img, format='png', bbox_inches='tight')
      self.image(tag, img, step)
  
  def log_image(self, tag, img, step=1, title='', wandb=True, tb=True):
    if FLAGS.write_summary and tb:
      self.image(tag, gezi.plot.tobytes(img), step, title)
    if FLAGS.wandb and wandb and FLAGS.wandb_image:
      import wandb
      wandb.log({tag: wandb.Image(img, caption=title)})
      
  # TODO for non tf
  def history(self, tag, values, step=1, bins=1000):
    """Log a histogram of the tensor of values."""

    # Create a histogram using numpy
    counts, bin_edges = np.histogram(values, bins=bins)

    # Fill the fields of the histogram proto
    hist = tf.compat.v1.HistogramProto()
    hist.min = float(np.min(values))
    hist.max = float(np.max(values))
    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values**2))

    # Drop the start of the first bin
    bin_edges = bin_edges[1:]

    # Add bin edges and counts
    for edge in bin_edges:
      hist.bucket_limit.append(edge)
    for c in counts:
      hist.bucket.append(c)

    # Create and write Summary
    summary = tf.compat.v1.Summary(
        value=[tf.compat.v1.Summary.Value(tag=tag, histo=hist)])
    self.writer.add_summary(summary, step)
    self.flush()

  def graph(self, model, input_to_model=None, verbose=False):
    return self.writer.add_graph(model, input_to_model, verbose)

  def embedding(self, mat, metadata=None, **kwargs):
    return self.writer.add_embedding(mat, metadata, **kwargs)

  def pr_curve(self, tag, labels, predictions, global_step=None, **kwargs):
    return self.writer.add_pr_curve(tag, labels, predictions, global_step,
                                    **kwargs)
