# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""CIFAR-10 data set.

See http://www.cs.toronto.edu/~kriz/cifar.html.
"""
import os

import tensorflow as tf
from absl import flags
FLAGS = flags.FLAGS

import gezi
import melt
logging = gezi.logging

import subprocess
import random
import numpy as np
import traceback
# from tfrecord_lite import decode_example 

from tensorflow.python.framework.sparse_tensor import SparseTensor

# NOTICE if batc_parse, first batch then map.. you have faster speed but each batch is the same size like 256 even for the last batch with drop remind=False
# have tested textlinedataset behavior like above TODO check tfrecord
class Dataset(object):
  def __init__(self, 
               subset='valid',
               batch_size=None,
               Type=None, 
               files=None,
               num_instances=None,
               batch_parse=None,
               sparse_to_dense=None,
               hvd_shard=True,
               use_int32=True,
               is_info=False,
               eval_keys=[],
               incl_keys=[],
               excl_keys=[],
               str_keys=[],
               use_tpu=False,
               recount=None):
    self.subset = subset
    self.filter_fn = None
    self.pos_filter_fn = None
    self.neg_filter_fn = None 
    self.count_fn = None
    self.Type = Type
    self.batch_parse = batch_parse if batch_parse is not None else FLAGS.batch_parse
    self.sparse_to_dense = sparse_to_dense if sparse_to_dense is not None else FLAGS.sparse_to_dense
    self.use_post_decode = None
    # if self.batch_parse:
    #   self.sparse_to_dense = False
    self.batch_size = batch_size or FLAGS.batch_size
    self.hvd_shard = hvd_shard
    self.indexes = {'train': -1, 'valid': -1, 'test': -1}
    self.is_info = is_info
    self.eval_keys = eval_keys or FLAGS.eval_keys
    if subset == 'test':
      self.eval_keys = gezi.get('test_keys') or self.eval_keys
    logging.debug('eval_keys in Dataset', self.eval_keys)
    self.show_keys = set()  # 如果用户不指定eval_keys 可以用self.show_keys所有非变成以及长度为0,1的key 前提需要使用.adds不能自己外部定义
    self.excl_keys = excl_keys or FLAGS.excl_keys
    self.incl_keys = incl_keys or FLAGS.incl_keys
    self.str_keys = str_keys

    self.parse_fn = tf.io.parse_single_example if not self.batch_parse else tf.io.parse_example

    self.features_dict = {}
    self.has_varlen_feats = False
    self.use_tpu = use_tpu
    try:
      # TPU detection. No parameters necessary if TPU_NAME environment variable is
      # set: this is always the case on Kaggle.
      tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
      # print('Running on TPU ', tpu.master())
    except ValueError:
      tpu = None
    if tpu is not None:
      self.use_tpu = True
    self.use_int32 = use_int32
    if self.use_tpu:
      self.use_int32 = True

    self.num_instances_ = num_instances
    self.files_ = files

    # self.use_post_decode = use_post_decode
    self.recount = recount or FLAGS.recount_tfrecords

    assert self.subset in ['train', 'valid', 'test'], \
          'subset is {} but should in [train, valid, test]'.format(self.subset)

  # TODO here just for specific online learning dataset
  @staticmethod
  def get_online_learning_files_24hours(span=2):
    dirs = FLAGS.train_input.split(',')
    dirs = [x for x in dirs if x and os.path.isdir(x)]
    files = gezi.list_files(dirs[-1])
    dirs = dirs[:-1]
    for i in range(len(dirs)):
      files_ = gezi.list_files(dirs[i])
      start = i * span 
      files_ = files_[start: start + span]
      files += files_

    return files

  def get_dir(self, subset=None):
    subset = subset or self.subset
    if subset == 'train':
      return FLAGS.train_input
    elif subset == 'valid' or subset == 'dev':
      return FLAGS.valid_input
    elif subset == 'test' or subset == 'infer':
      return FLAGS.test_input
    return None

  def get_online_learning_files_curhours():
    dirs = FLAGS.train_input.split(',')
    dirs = [x for x in dirs if x and os.path.isdir(x)]
    files = gezi.list_files(dirs[0])
    dirs = dirs[1:]
    parts = len(dirs)
    for i in range(len(dirs)):
      files_ = gezi.list_files(dirs[i])
      total = len(files_)
      span = int(total / parts)
      start = i * span 
      files_ = files_[start: start + span]
      files += files_

    return files

  @staticmethod
  def get_train_files():
    if FLAGS.dataset_mode == 'online24':
      return Dataset.get_online_learning_files_24hours()
    elif FLAGS.dataset_mode == 'onlinecur':
      return Dataset.get_online_learning_files_curhours()
    else:
      return []

  @staticmethod
  def get_filenames_(subset=None, shuffle=False):
    try:
      if subset in ['train', 'valid', 'test']:
        if subset == 'train':
          if not FLAGS.dataset_mode:
            files = gezi.list_files(FLAGS.train_input)
            if not files and FLAGS.train_input:
              files = gezi.list_files(FLAGS.train_input.split('|')[0])
          else:
            files = Dataset.get_train_files()
        elif subset == 'valid':
          files = gezi.list_files(FLAGS.valid_input)
        elif subset == 'test':
          files = gezi.list_files(FLAGS.test_input)
        if shuffle:
          np.random.shuffle(files)
        return files
      else:
        raise ValueError('Invalid data subset "%s"' % subset)
    except Exception:
      return None

  def get_filenames(self, subset=None, shuffle=False):
    subset = subset or self.subset
    return Dataset.get_filenames_(subset, shuffle=False)

  def basic_parse(self, example):
    self.auto_parse(keys=self.incl_keys, exclude_keys=self.excl_keys)
    fe = self.parse_(serialized=example)
    melt.try_append_dim(fe)
    return fe
  
  # override this
  def parse(self, example):
    return self.basic_parse(example)

  def decode(self, example):
    l = self.parse(example)
    
    if isinstance(l, (list, tuple)):
      features = l[0]
    else:
      features = l
    # self.use_tpu = True
    if isinstance(features, dict):
      if self.use_tpu:
        def decode_label(label):
          label = tf.io.decode_raw(label, tf.uint8)  # tf.string -> [tf.uint8]
          label = tf.reshape(label, [])  # label is a scalar
          return tf.cast(label, tf.int32) 
        # 目前，TPU 仅支持 tf.float32、tf.int32、tf.bfloat16 和tf.bool 数据类型。其他常见数据类型（例如 tf.uint8、tf.string 和 tf.int64）必须在数据预处理期间
        #（即在 TPUEstimator 的 input_fn 中）转换为某种受支持的数据类型。
        for key in features.keys():
          if features[key].dtype in [tf.int64, tf.uint8, tf.uint16, tf.uint32]:
            features[key] = tf.cast(features[key], tf.int32)
          # if features[key].dtype ==tf.string:
          #   features[key] = decode_label(features[key])
        # 注意valid和test还保留string 方便循环验证 但是model.call 需要额外mt.hack_tpu(input)处理一下保证model.predict的时候是去掉string的
        # if self.subset == 'train':
        if not self.is_info:
          keys = list(features.keys())
          for key in keys:
            if features[key].dtype ==tf.string:
              del features[key]

              if key in self.eval_keys:
                FLAGS.use_info_dataset = True  # 因为训练model的dataset不再包含eval的某个信息 需要依赖再遍历一遍info_dataset
              # features[key] = tf.ones_like(features[key], tf.int32)
              # features[key] = decode_label(features[key]) ## not work TODO

            logging.debug('de key', key, 'use_info_dataset', FLAGS.use_info_dataset)
      else:
        def _cast_dict(features):
          for key in features:
            if isinstance(features[key], dict):
              _cast_dict(features[key])
            else:
              if features[key].dtype == tf.int64 and self.use_int32:
                features[key] = tf.cast(features[key], tf.int32)
        _cast_dict(features)
 
      # is_info 只在tf2 keras模式下生效, 都创建 但是有可能不用 只有 FLAGS.use_info_dataset = True 才使用
      if self.is_info:
        keys = list(features.keys())
        if not FLAGS.predict_on_batch:
          if not self.eval_keys:
            for key in keys:
              dim = 1 if self.batch_parse else 0
              if not (len(features[key].shape) == dim or features[key].shape[dim] == 1):
                del features[key]
              else:
                self.show_keys.add(key)
          else:
            for key in keys:
              if key not in self.eval_keys:
                del features[key]
      else:
        keys = list(features.keys())
        for key in keys:
          if key in self.excl_keys:
            del features[key]

    return l

  def adjust(self, result):
    return result

  # # to be overrided if needed, 输入是已经batch好的数据
  # def post_decode(self, x, y):
  #   return x, y

  def parse_(self, serialized, features=None):
    ## 这样会去掉 (512, 0) 类似这样shape的key 他们是没有输入任何内容的varlen结果
    # for key in list(features.keys()):
    #   if key not in self.example:
    #     logging.debug('bad key in features, delete it not to parse:', key)
    #     del features[key]
    features = features or self.features_dict
    # ic(features)
    features = self.parse_fn(serialized=serialized, features=features)
    # print('------------------------------------------------features1', features)
    # features['subset'] = self.subset
    ## 这里强制去掉varlen 使用时候最好还是输入就不加入 varlen features
    if FLAGS.exclude_varlen_keys:
      sparse_keys = [key for key in features if isinstance(key, SparseTensor)]
      for key in sparse_keys:
        del features[key]
    else:
      if self.sparse_to_dense:
        modified = melt.sparse2dense(features, default_value=FLAGS.padding_idx)
        self.has_varlen_feats = modified
        # FLAGS.static_input = not modified # static_input不要被改变 因为后续可能py_function转换处理并设置固定长度输出
    self.features = features
    logging.debug(features)
    return features
  
  def gen_example(self, files=None):
    if not files:
      files = self.get_filenames()
    if not isinstance(files, (list, tuple)):
      files = [files]
    example = {}
    if files:
      for file in files:
        try:
          example = melt.first_example(file)
          logging.debug('example keys:', example.keys())
        except Exception:
          logging.error(traceback.format_exc())
          logging.error('bad tfrecord:', file)
        if example:
          self.example = example
          break
    self.example = example
    gezi.set('example', example)
    gezi.set('dataset_example', example)
    assert self.example, f'subset:{self.subset} dir:{self.get_dir()}\nfiles:{files}'
    return example

  def gen_input(self, files=None):
    example = self.gen_example().copy()
    for key in example:
      example[key] = np.asarray([example[key]])
    return example

  def first_input(self, files=None):
    return self.gen_input(files)

  def add(self, key, dtype=None, length=None, features_dict=None):
    features_dict = features_dict or self.features_dict
    dtype_ = dtype
    if key in self.example:
      dtype = dtype_ or self.example[key].dtype 
      if length is None:
        features_dict[key] = tf.io.VarLenFeature(dtype)
      elif length > 0:
        features_dict[key] = tf.io.FixedLenFeature([length], dtype)
      else:
        features_dict[key] = tf.io.FixedLenFeature([], dtype)
    
  def adds(self, keys, dtype=None, length=None, features_dict=None):
    features_dict = features_dict or self.features_dict
    dtype_ = dtype
    for key in keys:
      if key in self.example:
        dtype = dtype_ or self.example[key].dtype 
        if length is None:
          features_dict[key] = tf.io.VarLenFeature(dtype)
        elif length > 0:
          features_dict[key] = tf.io.FixedLenFeature([length], dtype)
        else:
          features_dict[key] = tf.io.FixedLenFeature([], dtype)

  def auto_parse(self, keys=[], exclude_keys=[], features_dict=None):
    keys = keys or FLAGS.dataset_keys or self.example.keys()
    exclude_keys = exclude_keys or FLAGS.dataset_excl_keys
    keys = [key for key in keys if key not in exclude_keys]

    for key in keys:
      if key not in self.example:
        continue
      length = self.example[key].shape[0]
      
      if length == 1:
        # just to (bs,), tf keras will auto change to (bs,1), also for string 0 is ok
        length = 0 

      dtype = melt.npdtype2tfdtype(self.example[key].dtype)
      # print(key, dtype, length, self.example[key])
      self.adds([key], dtype, length, features_dict)

  def adds_varlens(self, keys=[], exclude_keys=[], features_dict=None):
    keys = keys or self.example.keys()
    keys = [key for key in keys if key not in exclude_keys]

    for key in keys:
      if not key in self.example:
        continue
      length = self.example[key].shape[0]
      dtype = melt.npdtype2tfdtype(self.example[key].dtype)
      length = None
      if dtype == tf.string:
        length = 1
      self.adds([key], dtype, length, features_dict)  

  # {'abc': tf.keras.layers.Input(shape, dtype), ..}
  def get_inputs(self):
    if not self.features:
      self.make_batch()
    # return melt.features2inputs(self.features)
    if FLAGS.batch_parse:
      return melt.features2inputs(self.features)
    else:
      # 但是这样有风险比如example里面是图像 dataset读取的时候处理为float  TODO
      return melt.example2inputs(self.example, self.features.keys())
  
  def make_batch(self, 
                 batch_size=None, 
                 filenames=None,
                 subset=None,
                 initializable=False,
                 repeat=None,
                 shuffle=None,
                 return_iterator=True,
                 hvd_shard=None,
                 simple_parse=False,
                 num_epochs=None,
                 cache=False,
                 cache_file='',
                 buffer_size=None,
                 batch_sizes=None,
                 buckets=None,
                 drop_remainder=None,
                 world_size=1,
                 rank=0,
                 shard_by_files=None,
                 distribute_strategy=None,
                 return_numpy=False):
    """Read the images and labels from 'filenames'."""
    # with tf.device('/cpu:0'):
    subset = subset or self.subset
    hvd_shard = hvd_shard if hvd_shard is not None else self.hvd_shard
    if batch_size is None:
      is_test = True
    else:
      is_test = False
    batch_size = batch_size or self.batch_size
    self.batch_size = batch_size
    batch_sizes = batch_sizes if batch_sizes is not None else FLAGS.batch_sizes
    buffer_size = buffer_size if buffer_size is not None else FLAGS.buffer_size
    buckets = buckets if buckets is not None else FLAGS.buckets
    drop_remainder = drop_remainder if drop_remainder is not None else FLAGS.drop_remainder
    shard_by_files = shard_by_files if shard_by_files is not None else FLAGS.shard_by_files
    # use_post_decode = use_post_decode if use_post_decode is not None else self.use_post_decode

    self.return_numpy = return_numpy

    filenames = filenames or self.files_ or self.get_filenames(subset)
    
    self.gen_example(filenames)

    is_eager = tf.executing_eagerly()

    logging.debug(subset, 'num files', len(filenames))
    assert filenames,  f'{subset}:{filenames}  train:{FLAGS.train_input}, valid:{FLAGS.valid_input}, test:{FLAGS.valid_input}' 

    self.files_ = filenames

    self.indexes[self.subset] += 1
    
    if repeat is None:
      num_gpus = melt.num_gpus() if not 'OMPI_COMM_WORLD_RANK' in os.environ else 1
      # if subset == 'train' or num_gpus > 1:
      if subset == 'train':
        repeat = True
      else:
        repeat = False
      if is_eager and num_gpus == 1 and tf.__version__ < '2':
        # let tf eager similary to pytorch
        repeat = False

    if shuffle is None:
      if subset == 'train':
        shuffle = FLAGS.shuffle 
      else:
        shuffle = FLAGS.shuffle_valid 

    if drop_remainder is None:
      if gezi.get('tpu'):
        drop_remainder = True
      else:
        if subset == 'train':
          drop_remainder = True
        else:
          drop_remainder = False

    balance_pos_neg=False
    if self.pos_filter_fn and self.neg_filter_fn:
      balance_pos_neg = True

    if self.subset != 'train' and FLAGS.eval_batch_size:
      batch_sizes = None
      buckets = None
    else:
      if buckets:
        buckets = [int(x) for x in buckets]
        FLAGS.buckets = buckets
      if batch_sizes:
        batch_sizes = [int(x) for x in batch_sizes]
        if batch_sizes[0] < batch_size:
          factor = batch_size / batch_sizes[0]
          batch_sizes = [int(x * factor) for x in batch_sizes]
        FLAGS.batch_sizes = batch_sizes

    # repeat = False
    logging.debug('---dataset subset:', self.subset, 'repeat:', repeat, 'batch_parse:', self.batch_parse, 
                  'drop_last:', drop_remainder, 'initializable:', initializable, 'shuffle:', shuffle,
                  'wolrd_size', world_size, 'rank', rank, 'batch_size', batch_size)

    seed = FLAGS.seed 
    if seed is not None:
      FLAGS.seed += 1

    logging.debug(f'seed for {self.subset} dataset is {seed}')

    ## put on cpu or dummy
    with melt.device(FLAGS.dataset_device):
      result = melt.dataset_decode.inputs(
        filenames, 
        decode_fn=self.decode,
        batch_size=batch_size,
        post_decode_fn=self.post_decode if hasattr(self, 'post_decode') and self.use_post_decode != False else None,
        shuffle=shuffle,
        shuffle_batch=FLAGS.shuffle_batch,
        shuffle_files=FLAGS.shuffle_files,
        ordered=FLAGS.dataset_ordered if subset == 'train' else True,
        num_threads=FLAGS.num_threads,
        buffer_size=buffer_size,
        num_prefetch_batches=FLAGS.num_prefetch_batches,
        initializable=initializable,
        repeat=repeat,
        repeat_then_shuffle=FLAGS.repeat_then_shuffle,
        drop_remainder=drop_remainder,
        bucket_boundaries=buckets,
        bucket_batch_sizes=batch_sizes,
        length_index=FLAGS.length_index,
        length_key=FLAGS.length_key,
        seed=seed,
        return_iterator=return_iterator,
        filter_fn=self.filter_fn,  # inside filter_fn judge subset train or valid or test
        balance_pos_neg=balance_pos_neg,
        pos_filter_fn=self.pos_filter_fn if subset == 'train' else None,
        neg_filter_fn=self.neg_filter_fn if subset == 'train' else None,
        count_fn=self.count_fn if subset == 'train' else None,
        name=subset,
        Dataset=self.Type,
        batch_parse=self.batch_parse,
        hvd_shard=hvd_shard,
        shard_by_files=shard_by_files,
        training=subset == 'train',
        simple_parse=simple_parse,
        num_epochs=num_epochs,
        dynamic_pad=FLAGS.dynamic_pad, #如果有varlen feats才需要 padded_batch 同时batch_parse模式其实也不需要因为sparse2dense就可以自动padd
        cache=cache,
        cache_file=cache_file,
        device='/gpu:0',
        world_size=world_size,
        rank=rank,
        fixed_random=FLAGS.fixed_random,
        parallel_read_files=FLAGS.parallel_read_files,
        use_feed_dict=FLAGS.train_loop and FLAGS.rounds > 1 and not is_eager and FLAGS.feed_dataset and tf.__version__ < '2',
        feed_name=f'{self.subset}_{self.indexes[self.subset]}' if not is_test else None,
        padding_values=FLAGS.padding_idx, 
        distribute_strategy=distribute_strategy or melt.distributed.get_strategy(),
        torch=FLAGS.torch,
        keras=FLAGS.keras,
        subset=self.subset,
        return_numpy=return_numpy,
        ) 
      
    result = self.adjust(result)
    return result
    
  @staticmethod
  def num_examples_per_epoch(subset, dir=None):
    def _get_num_records(dir):
      if not tf.io.gfile.exists(dir):
        logging.warning(f'{dir} not exist and return 0 num records')
        return 0
      file = os.path.join(dir, 'num_records.txt')
      num_examples = 0
      if not FLAGS.recount_tfrecords:
        num_examples = gezi.read_int_from(file, 0)
      if not num_examples:
        num_examples = melt.get_num_records('%s/*.tfrec' % dir)
        # gezi.write_to_txt(num_examples, file)
      return num_examples
    default_value = None
    num_examples = 0
    dir_count = 0
    if subset == 'train':
      if not FLAGS.dataset_mode:
        if ',' not in FLAGS.train_input:
          dir = dir or gezi.dirname(FLAGS.train_input)
          num_examples = _get_num_records(dir)
          dir_count = 1
        else:
          num_examples = 0
          for train_input in FLAGS.train_input.strip().split(','):
            if train_input:
              dir = gezi.dirname(train_input) if not tf.io.gfile.isdir(train_input) else train_input
              num_examples_indir = _get_num_records(dir)
              num_examples += num_examples_indir
              if num_examples_indir: 
                logging.debug(f'train: {dir} files:{len(gezi.list_files(dir))} samples:{num_examples_indir}')
                dir_count += 1
    elif subset == 'valid':
      if not FLAGS.valid_input:
        return 0
      if ',' not in FLAGS.valid_input:
        dir = dir or gezi.dirname(FLAGS.valid_input)
        num_examples = _get_num_records(dir)
        dir_count = 1
      else:
        num_examples = 0
        for valid_input in FLAGS.valid_input.strip().split(','):
          if valid_input:
            dir = gezi.dirname(valid_input) if not tf.io.gfile.isdir(valid_input) else valid_input
            num_examples_indir = _get_num_records(dir)
            num_examples += num_examples_indir
            if num_examples_indir: 
              logging.debug(f'{dir}:{num_examples_indir}')
              dir_count += 1
    elif subset == 'test':
      if not FLAGS.test_input:
        return 0
      if ',' not in FLAGS.test_input:
        dir = dir or gezi.dirname(FLAGS.test_input)
        num_examples = _get_num_records(dir)
        dir_count = 1
      else:
        for test_input in FLAGS.teset_input.strip().split(','):
          if test_input:
            dir = gezi.dirname(test_input) if not tf.io.gfile.isdir(test_input) else test_input
            num_examples += _get_num_records(dir)        
    else:
      raise ValueError('Invalid data subset "%s"' % subset)

    logging.debug(subset, 'dir_count', dir_count)
    
    if not num_examples:
      logging.debug(f'Could not find num_records.txt and not set num examples so count for {subset}')
      files = Dataset.get_filenames_(subset)
      try:
        num_examples = melt.get_num_records(files)
      except Exception:
        files_str = ' '.join(files)
        num_examples = int(subprocess.check_output(f"cat {files_str} | wc -l ", shell=True).split()[0]) 
      if not dir:
        if subset == 'train' and os.path.isdir(FLAGS.train_input):
          dir = FLAGS.train_input 
        if subset == 'valid' and os.path.isdir(FLAGS.valid_input):
          dir = FLAGS.valid_input  
        if subset == 'test' and os.path.isdir(FLAGS.test_input):
          dir = FLAGS.test_input
      if dir:   
        file = os.path.join(dir, 'num_records.txt')
        logging.debug(f'write {num_examples} to {file}')
        # gezi.write_to_txt(num_examples, file)
      assert num_examples
      
    assert not (subset == 'train' and FLAGS.min_train and num_examples < FLAGS.min_train), '%d %d' % (num_examples, FLAGS.min_train)
    assert not (subset == 'valid' and FLAGS.min_valid and num_examples < FLAGS.min_valid), '%d %d' % (num_examples, FLAGS.min_valid)

    return num_examples

  @staticmethod
  def num_examples(subset, dir=None):
    return Dataset.num_examples_per_epoch(subset, dir)

  @property
  def num_instances(self):
    if self.num_instances_:
      return self.num_instances_
    assert self.files_
    self.num_instances_ = melt.get_num_records(self.files_, recount=self.recount)
    return self.num_instances_

  @property
  def files(self):
    return self.files_

  @property
  def records(self):
    return self.files_

  def __len__(self):
    return self.num_instances or Dataset.num_examples_per_epoch(self.subset)

  @property
  def num_steps(self):
    return -(-len(self) // self.batch_size)

