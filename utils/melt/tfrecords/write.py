#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   write.py
#        \author   chenghuige  
#          \date   2016-08-24 10:21:46.629992
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import random
import numpy as np
import tensorflow as tf
import gezi
import melt
 
class Writer(object):
  def __init__(self, filename, format='tfrec', buffer_size=None, 
               shuffle=True, seed=None, clear_first=False):
    '''
    buffer_size = None means write at once
    = 0 means buffersize large engouh, only output at last 
    oterwise output when buffer full
    '''
    if seed:
      random.seed(seed)
    self.count = 0
    self.buffer_size = buffer_size
    self.shuffle = shuffle
    
    assert filename.endswith('.' + format), f'file:{filename} format:{format}'
    # filename = filename.replace('.' + format, '.TMP')
    filename_ = filename[:-len(format)-1]
    filename = filename_ + '.TMP'
    dir_ = os.path.dirname(filename)
    os.makedirs(dir_, exist_ok=True)

    # TODO problem here might cause data loss, delete by yourself manualy which is safe
    if clear_first:
      command = f'rm -rf {dir_}/{filename_}.*.{format}'
      ic(command)
      os.system(command)
    
    self.writer = tf.io.TFRecordWriter(filename)
    self.buffer = [] if self.buffer_size else None
    self.sort_vals = []

    self.filename = filename
    self.format = format

    self.closed = False

  def __del__(self):
    #print('del writer', file=sys.stderr)
    self.close()

  def __enter__(self):
    return self  

  def __exit__(self, exc_type, exc_value, traceback):
    #print('close writer', file=sys.stderr)
    self.close()

  def close(self):
    if not self.closed:
      if self.buffer:
        if self.shuffle:
          random.shuffle(self.buffer)
        for example in self.buffer:
          self.writer.write(example.SerializeToString())
        self.buffer = []  
        self.sort_vals = []

      ifile = self.filename 
      if self.num_records:
        ofile = ifile[:-len('.TMP')] + f'.{self.num_records}.{self.format}'
        os.rename(ifile, ofile)
      else:
        print(f'removing {ifile}')
        gezi.try_remove(ifile)
      self.closed = True
      self.count = 0
    
  def finalize(self):
    self.close()
    
  def write(self, feature, sort_val=None):
    self.write_feature(feature, sort_val)

  def write_feature(self, feature, sort_key=None):
    fe = melt.gen_features(feature)
    example = tf.train.Example(features=tf.train.Features(feature=fe))
    if sort_key is None:
      self.write_example(example)
    else:
      self.write_example(example, feature[sort_key])

  def write_example(self, example, sort_val=None):
    self.count += 1
    if self.buffer is not None:
      self.buffer.append(example)
      if sort_val is not None:
        self.sort_vals.append(sort_val)
      if len(self.buffer) >= self.buffer_size and self.buffer_size != 0:
        if self.sort_vals:
          assert self.buffer_size == 0, 'sort all values require buffer_size==0'
          yx = zip(self.sort_vals, self.buffer)
          yx.sort()
          self.buffer = [x for y, x in yx]
        elif self.shuffle: # if sort_vals not do shuffle anymore
          random.shuffle(self.buffer)
        for example in self.buffer:
          self.writer.write(example.SerializeToString())
        self.buffer = []
    else:
      self.writer.write(example.SerializeToString())

  def size(self):
    return self.count

  @property
  def num_records(self):
    return self.count

class MultiWriter(object):
  """
  sequence read and output to mutlitple tfrecord
  """
  def __init__(self, dir, max_records, format='tfrec'):
     self.dir = dir
     self.max_records = max_records
     self.index = 0
     self.count = 0
     self.format = format
     
     self.writer = self.get_writer()
  
  def __del__(self):
    # print('del writer', file=sys.stderr)
    self.close()

  def __enter__(self):
    return self  

  def __exit__(self, exc_type, exc_value, traceback):
    # print('close writer', file=sys.stderr)
    self.close()

  def get_writer(self):
    return Writer(f'{self.dir}/{self.index}.{self.format}')
  
  def write_feature(self, feature):
    feature = melt.gen_features(feature)
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    self.write(example)

  def write(self, example):
    self.writer.write(example)
    self.count += 1
    if self.count == self.max_records:
      self.index += 1
      self.writer.close()
      self.writer = self.get_writer()
      self.count = 0

  def close(self):
    self.writer.close()

class RandomSplitWriter(object):
  """
  read single file, random split as train, test to two files
  """
  def __init__(self, train_file, test_file, train_ratio=0.8):
    self.train_writer = Writer(train_file)
    self.test_writer = Writer(test_file)
    self.train_ratio = train_ratio

  def __enter__(self):
    return self  

  def __del__(self):
    print('del writer', file=sys.stderr)
    self.close()

  def __exit__(self, exc_type, exc_value, traceback):
    print('close writer', file=sys.stderr)
    self.close()
    
  def close(self):
    self.train_writer.close()
    self.test_writer.close()

  def write(example):
    writer = self.train_writer if np.random.random_sample() < self.train_ratio else self.test_writer()
    writer.write(example)

class RandomSplitMultiOutWriter(object):
  """
  read single file, random split as train, test each to mulitple files
  """
  def __init__(self, train_dir, test_dir, train_name='train', test_name='test', max_lines=50000, train_ratio=0.8):
    self.train_writer = MultiOutWriter(train_dir, train_name, max_lines)
    self.test_writer = MultiOutWriter(test_dir, test_name, max_lines)
    self.train_ratio = train_ratio

  def __enter__(self):
    return self  

  def __del__(self):
    print('del writer', file=sys.stderr)
    self.close()

  def __exit__(self, exc_type, exc_value, traceback):
    print('close writer', file=sys.stderr)
    self.close()

  def close(self):
    self.train_writer.close()
    self.test_writer.close()

  def write(self, example):
    writer = self.train_writer if np.random.random_sample() < self.train_ratio else self.test_writer()
    writer.write(example)

