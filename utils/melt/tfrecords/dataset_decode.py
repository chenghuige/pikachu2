#!/usr/bin/env python
#coding=utf8
# ==============================================================================
#          \file   read_sparse.py
#        \author   chenghuige  
#          \date   2016-08-15 20:13:06.751843
#   \Description  @TODO https://github.com/tensorflow/tensorflow/tree/r0.10/tensorflow/contrib/slim/python/slim/data/
# ==============================================================================
 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from absl import flags
FLAGS = flags.FLAGS

import gezi
import melt
logging = gezi.logging

import sys
import os
import numpy as np
import inspect
import traceback

def inputs(files, 
           decode_fn, 
           batch_size=64,
           post_decode_fn=None,
           num_epochs = None, 
           num_threads=None, 
           buffer_size = 15000, #change from 1000 to 15000
           dynamic_pad=True,
           shuffle=True,
           shuffle_batch=None,
           shuffle_files=None,
           ordered=None,
           min_after_dequeue=None, #depreciated
           seed=None, 
           enqueue_many=False,  #depreciated
           fixed_random=False, 
           drop_remainder=False, 
           num_prefetch_batches=None, 
           bucket_boundaries=None,
           length_index=None,
           length_key=None,
           length_fn=None,
           bucket_batch_sizes=None,
           repeat=True,
           initializable=False,
           filter_fn=None,
           balance_pos_neg=False,
           pos_filter_fn=None,
           neg_filter_fn=None,
           count_fn=None,
           return_iterator=False,
           Dataset=None,
           batch_parse=False, #by default will be line parse
           hvd_shard=True,
           shard_by_files=False,
           training=False,
           simple_parse=False,
           repeat_then_shuffle=False,
           cache=False,
           cache_file='',
           device=None,
           world_size=1,
           rank=0,
           parallel_read_files=False,
           use_feed_dict=False,
           feed_name=None,
           padding_values=None,
           distribute_strategy=None,
           torch=False,
           keras=False,
           subset=None,
           return_numpy=False,
           name='input'):
  """Reads input data num_epochs times.
  for sparse input here will do:
  1. read serialized_example
  2. shuffle serialized_examples
  3. decdoe batch_serialized_examples
  notice read_sparse.inputs and also be used for dense inputs,but if you 
  only need to decode part from serialized_example, then read.inputs will 
  be better, less to put to suffle
  #--------decode example, can refer to libsvm-decode.py
  # def decode(batch_serialized_examples):
  #   features = tf.parse_example(
  #       batch_serialized_examples,
  #       features={
  #           'label' : tf.FixedLenFeature([], tf.int64),
  #           'index' : tf.VarLenFeature(tf.int64),
  #           'value' : tf.VarLenFeature(tf.float32),
  #       })

  #   label = features['label']
  #   index = features['index']
  #   value = features['value']

  #   return label, index, value 

  #string_input_reducer will shuffle files
  #shuffle will read file by file and shuffle withn file(in shuffle queue) 
  #shuffle_batch_join will read multiple files and shuffle in shuffle queue(from many files)

  To get fixed sequence 
  shuffle=False  so by this way the sequence is as your data input unchange
  or
  shuffle=True
  seed=1024 #set
  batch_join=False  by this way you have fixed random, so get same result
  NOTICE, shuffle=True,seed=1024,batch_join=True will not get same result
  shuffle=False,seed=1024,batch_join=True also, so batch_join seems seed only control inqueue random, can not get fixed result

  for no random -> fixed result set shuffle=False wihch will force batch_join=False then use batch
  for fixed random ->  shuffle=True, seed set or  fix_random=True
  read-records.py show above ok, but train-evaluate.py show not, only shuffle=False can get fixed result.. @FIXME strange
  for train-evaluate.py it looks you can set shuffle in string_input_producer True, but then must use batch,
  batch_join and shuffle_batch join all not fixed even with seed set, may be due to trainset two inputs read ?
  for read-records.py batch_join will be fixed, shuffle_batch_join not 

  defualt parmas will give max random...

  Args:
  decode: user defined decode 
  min_after_dequeue: set to >2w for production train, suggesed will be 0.4 * num_instances, but also NOTICE do not exceed mem
  #--default parmas will make most randomness
  shuffle_files: wehter shuffle file 
  shuffle_batch: batch or shuffle_batch
  batch_join: wether to use multiple reader or use one reader mutlitple thread
  fix_random: if True make at most random which can fix random result
  allow_smaller_final_batch: set True usefull if you want verify on small d

  great article http://d0evi1.com/tensorflow/ds_performance/
  https://www.tensorflow.org/versions/master/performance/ds_performance
  """
  Dataset = Dataset or tf.data.TFRecordDataset
  AUTO = tf.data.experimental.AUTOTUNE

  # if filter_fn or pos_filter_fn or neg_filter_fn:
  #   batch_parse = False

  # repeat_then_shuffle = True

  # if bucket_boundaries or filter_fn or pos_filter_fn or neg_filter_fn or neg_filter_fn:
  #   assert not batch_parse, 'try FLAGS.batch_parse=0'
  
  use_horovod = 'OMPI_COMM_WORLD_RANK' in os.environ
  if use_horovod:
    if FLAGS.torch:
      import horovod.torch as hvd
    else:
      import horovod.tensorflow as hvd

  def shard(d):
    return d.shard(hvd.size(), hvd.rank())

  # Choose to use cpu outside input function like in dataset.py
  #with tf.device('/cpu:0'):
  if isinstance(files, str):
    files = gezi.list_files(files)
  assert len(files) > 0

  # if use_horovod and not hvd_shard and training:
  if use_horovod and not hvd_shard:
    # assert len(files) % hvd.size() == 0, '{} {} {}'.format(len(files), files, hvd.size())
    files_ = []
    for i in range(len(files)):
      if i % hvd.size() == hvd.rank():
        files_.append(files[i])
    files = files_
    logging.info('----------train-files', files)
    #exit(0)

  if not num_threads:
    try:
      import multiprocessing
      #num_threads = int(multiprocessing.cpu_count() * 0.6)
      num_threads = multiprocessing.cpu_count() 
      logging.debug('num_threads as multiprocessing.cpu_count', num_threads)
    except Exception:
      num_threads = 12
      logging.debug('num_threads set by default', num_threads)

  if 'batch_size' in inspect.getargspec(decode_fn).args:
    decode_fn_ = decode_fn
    def decode_function(example):
      return decode_fn_(example, batch_size)
    decode_fn = decode_function

  # TODO simple parase is depreciated !

  if simple_parse:
    # for multiple gpu horovod run seem this much better, might due to repeat then shuffle better TODO 
    d = Dataset(files)
    if use_horovod and hvd_shard:
      d = shard(d)
    if repeat:
      d = d.repeat(num_epochs)
      if traning:
        d = d.shuffle(batch_size * 1024)
      d = d.batch(batch_size).map(decode_fn, num_parallel_calls=AUTO).prefetch(AUTO)
    else:
      if training:
        d = d.shuffle(batch_size * 1024, reshuffle_each_iteration=True)
        d = d.batch(batch_size).map(decode_fn, num_parallel_calls=AUTO).prefetch(AUTO)
        
    if tf.executing_eagerly() or tf.__version__ >= '2':
      return d
    else:
      if not initializable:
        return tf.compat.v1.data.make_one_shot_iterator(d)
      else:
        return tf.compat.v1.data.make_initializable_iterator(d)
    
  if not num_epochs: 
    num_epochs = None

  if shuffle:
    if shuffle_files is None:
      shuffle_files = True
    if shuffle_batch is None:
      shuffle_batch = True
  else:
    if shuffle_files is None:
      shuffle_files = False
    if shuffle_batch is None:
      shuffle_batch = False
    # TDO 并行读取就会打乱顺序？
    if not shuffle_files:
      parallel_read_files = False

  if fixed_random:
    if seed is None:
      seed = 1024
  else:
    # if shuffle_files:
    #   # melt.init np.random.seed已经绑定
    #   np.random.shuffle(files)
    pass

  num_files = len(files)
  if use_feed_dict and feed_name:
    files = tf.compat.v1.placeholder(tf.string, [None], feed_name)
    gezi.set_global(feed_name, files)

  if not num_prefetch_batches:
    #num_prefetch_batches = num_threads + 3
    if buffer_size:
      num_prefetch_batches = int(buffer_size / batch_size)
    # else:
    #   num_prefetch_batches = 100
  
  if not buffer_size and num_prefetch_batches:
    buffer_size = num_prefetch_batches * batch_size
    
  # single gpu from 0.37h to 0.31h...
  # https://github.com/alibaba/FastNN/blob/master/images/utils/dataset_utils.py
  # from tensorflow.contrib.data.python.ops import threadpool
  # d = threadpool.override_threadpool(
  #   d,
  #   threadpool.PrivateThreadPool(
  #     num_threads, display_name='input_pipeline_thread_pool'))

  options = tf.data.Options()
  try:
    options.threading.private_threadpool_size = num_threads
  except Exception:
    options.experimental_threading.private_threadpool_size = num_threads

  ## https://fantashit.com/dataset-sharding-in-multiworker-mirrored-strategy/
  ## 但是似乎无效 目前只能 --valid_interval_step = 0 不做val loss验证了 tf >= 2.4, tf 2.6rc2似乎修复了这个问题
  ## https://github.com/tensorflow/tensorflow/issues/45157
  options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
  # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

  options.experimental_deterministic = True

  if shuffle and not fixed_random:
    options.experimental_deterministic = False

  # 及时train要求不 shuffle 如果FLAGS.dataset_ordered!=True (默认) 也强制 不要求ordered 为了速度 牺牲一定 非shuffle的确定性 对train也还好
  # 如果要求完整的训练确定性 FLAGS.dataset_ordered = True
  if not ordered:
    options.experimental_deterministic = False

  logging.debug('dataset decode', subset, 'buffer_size:', buffer_size, 'shuffle:', shuffle,
                'shuffle_files:', shuffle_files, 'shuffle_batch:', shuffle_batch, 'fixed_random:', fixed_random,
                'ordered:', options.experimental_deterministic,
                'num_threads:', num_threads, 'world_size:', world_size)
  
  # with tf.compat.v1.name_scope(name): # to 485

  # https://github.com/tensorflow/tensorflow/issues/14857
  # for CloudS tested using parallel_read_files is much slower < 30it/s
  # shuffle_files from 75% then up, otherwise from 40% up to 70+it/s, but for other usecases parallel_read_files might help speedup
  # 👆原因是非sgsapp数据源的样本长度短 并行读取不同文件意味着更好的随机性 而反之则会有比如很多都是比较短的batch在后面出现 相应的训练的速度也会快 并不是读取变快
  # TODO tf.train.match_filenames_once 可以尝试这个对horovod shard
  # https://github.com/horovod/horovod/issues/249 How to split dataset by multi workers
  
  # TODO 现在似乎有问题 就是如果shard 而且shard by files = False, 同时是shuffle模式 走下面的parallel read files就会有问题
  # 似乎就是要么只能shuffle 不要并行读文件， 要么并行读文件 不要后续shuffle 。。。 当然如果文件数目均分可以 还是使用shard_by_files最好, 一般valid infer 倒是也不需要shuffle
  if (use_horovod or world_size > 1) and (not shard_by_files) and shuffle_batch:
    logging.warning('multiprocess shard not by files and do shuffle then force not to use paralle read files')
    parallel_read_files = False

  if not parallel_read_files or num_files == 1:
    d = Dataset(files)
    d = d.with_options(options)
    if use_horovod and hvd_shard:
      d = shard(d)
    if not use_horovod and world_size > 1:
      d = d.shard(world_size, rank)
  else:
    # Be sure to shard before you use any randomizing operator (such as shuffle).
    # Generally it is best if the shard operator is used early in the dataset pipeline. 
    # For example, when reading from a set of TFRecord files, shard before converting the dataset to input samples. 
    # This avoids reading every file on every worker. The following is an example of an efficient sharding strategy within a complete pipeline:
    try:
      # Note: The default behavior of this method is to return filenames in a non-deterministic random shuffled order. 
      # Pass a seed or shuffle=False to get results in a deterministic order.
      # https://zhuanlan.zhihu.com/p/92763981 坑 
      # 在于 `tf.data.Dataset.list_files` 这个API，返回的文件列表默认情况下是被打乱了的，而且不同worker上打乱的顺序还不一样，这样就导致后面再执行`d.shard` 分片操作的时候，
      # 各个worker所获得的数据子集之间并不遵循MCMC原则，也就是所有worker处理的数据加起来可能并不是整个数据集，同时有部分数据子集可能被多个worker重复处理了。
      if shffle_files and (use_horovod or world_size > 1):
        assert seed
      d = tf.data.Dataset.list_files(files, shuffle=shuffle_files, seed=seed)
      d = d.with_options(options)
      # d = d.shuffle(num_files)
    except Exception:
      d = tf.data.Dataset.from_tensor_slices(files)
      d = d.with_options(options)
    # here shard by files, not work good, especially for text line dataset with horovod
    if use_horovod and shard_by_files:
      logging.debug('shard by files')
      d = shard(d)
    elif world_size > 1 and shard_by_files:
      d = d.shard(world_size, rank)

    ## already random.shuffle
    # if shuffle_files:
    #   d = d.shuffle(num_files, seed=seed)
    ## interleave 和直接 TFRecordDataset(num_parallel_reads)应该等价
    # d = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files
    
    # 这个如果在shard之前 还是可能造成不符合MCMC 数据交叉重叠
    # if not ((use_horovod or world_size > 1) and (not shard_by_files)): #not work。。。Could not parse example input, value:
    if tf.__version__ < '1.15':
      d = d.interleave(Dataset,
                        cycle_length=min(num_files, 1000),  # in tf 1.14 must set and can not set as AUTOTUNE for tf 2.1 with default as AUTOTUNE
                        block_length=1,
                        num_parallel_calls=AUTO)
    else:
      d = d.interleave(Dataset,
                      #  cycle_length=min(len(files), 1000),  # in tf 1.14 must set and can not set as AUTOTUNE for tf 2.1 with default as AUTOTUNE
                      block_length=1,
                      num_parallel_calls=AUTO)
      
    #------不再被走到
    if use_horovod and not shard_by_files:
      logging.debug('not shard by files')
      # TODO FIXME shard here has problem if not tf.random.set_random_seed then here might each worker accese same data part, reproduce by projects/feed/rank/read/test-read2.py
      d = shard(d)
    elif world_size > 1 and not shard_by_files:
      d = d.shard(world_size, rank)

  # options = tf.data.Options()
  # options.experimental_optimization.noop_elimination = True
  # options.experimental_optimization.map_vectorization.enabled = True
  # options.experimental_optimization.apply_default_optimizations = False
  # d = d.with_options(options)

  if repeat and repeat_then_shuffle:
    d = d.repeat(num_epochs)

  # must batch then map if use pyfunc which you might use batch_parse, here batch_parse means batch parse otherwise slower but simple and powerfull...
  if not batch_parse:
    d = d.map(decode_fn, num_parallel_calls=AUTO)
    if cache:
      logging.debug('Cache dataset')
      d = d.cache(cache_file)

  ## CHECK  This utility method replaces the deprecated-in-V2 tf.compat.v1.Dataset.output_shapes property.
  #   In [8]: d._flat_shapes
  # Out[8]: [TensorShape([])]

  # In [9]: d._flat_types
  # Out[9]: [tf.int32]
  try:
    #shapes = d._output_shapes 
    shapes = tf.compat.v1.data.get_output_shapes(d)
    # shapes = d._flat_shapes
  except Exception:
    shapes = None

  # compat.v1接口后续如何维护 ？
  if dynamic_pad and shapes is not None and padding_values is not None:
    if not isinstance(padding_values, (list, tuple)):
      # TODO
      types =  tf.compat.v1.data.get_output_types(d)
      # types = d._flat_types  # not ok...The two structures don't have the same sequence length. Input structure has length 2, while shallow structure has length 10.
      padding_value = padding_values
      padding_values = [padding_value] * len(shapes)
      if len(shapes) == 2:
        m = {}
        for key in shapes[0]:
          if types[0][key] == tf.string:
            m[key] = ''
          else:
            m[key] = tf.cast(padding_value, types[0][key])
        padding_values[0] = m
        padding_values[1] = tf.cast(padding_value, types[1])
      padding_values = tuple(padding_values)

  logging.debug('datast decode shapes', shapes)
  logging.debug('padding_values', padding_values)
  
  ## Has bug.. seems as least not work with bucket not sure without bucket ok or not
  if balance_pos_neg:
    # https://stackoverflow.com/questions/46938530/produce-balanced-mini-batch-with-d-api/49283371#49283371
    ds_pos = d.filter(pos_filter_fn).repeat()
    ds_neg = d.filter(neg_filter_fn)

    # def _concat(x, y):
    #   return tf.cond(tf.random_uniform(()) > 0.5, lambda: x, lambda: y)
    # d = tf.data.Dataset.zip((ds_pos, ds_neg))
    # d = d.map(_concat)

    d = tf.data.Dataset.zip((ds_pos, ds_neg))
    # Each input element will be converted into a two-element `Dataset` using
    # `Dataset.from_tensors()` and `Dataset.concatenate()`, then `Dataset.flat_map()`
    # will flatten the resulting `Dataset`s into a single `Dataset`.
    d = d.flat_map(
        lambda ex_pos, ex_neg: tf.data.Dataset.from_tensors(ex_pos).concatenate(
            tf.data.Dataset.from_tensors(ex_neg)))

  #https://github.com/tensorflow/tensorflow/issues/14451
  # count_fn for over sample
  if count_fn is not None:
    d = d.flat_map(
      lambda x, y : tf.data.Dataset.from_tensors((x, y)).repeat(tf.cast(count_fn(x, y), dtype=tf.int64)))

  # filter fn for under sample
  # if under_sample_filter_fn is not None:
  #   d = d.filter(under_sample_filter_fn)
    
  if filter_fn is not None and not batch_parse:
    d = d.filter(filter_fn)

  if shuffle_batch:
    logging.debug('shuffle with buffer_size', buffer_size, 'seed', seed)
    d = d.shuffle(buffer_size=buffer_size, seed=seed, reshuffle_each_iteration=True)

  # shuffle then repeat
  if repeat and not repeat_then_shuffle:
    d = d.repeat(num_epochs)

  # #https://github.com/HKUST-KnowComp/R-Net/blob/master/util.py
  # #https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/data_reader.py
  
  if bucket_boundaries:
    shapes = None
    # def element_length_fn(example): return tf.size(example[0][length_key])
    # d = d.apply(
    #           tf.data.experimental.bucket_by_sequence_length(
    #               element_length_func=element_length_fn,
    #               bucket_batch_sizes=bucket_batch_sizes,
    #               bucket_boundaries=bucket_boundaries,
    #    ))
    # TODO remove support for length index, use use length key!
    assert length_key is not None or length_index is not None, 'forget to set length key  or length index ?'
    if not isinstance(bucket_boundaries, (list, tuple)):
      boundaries = [int(x) for x in bucket_boundaries.split(',') if x.strip()]
    else:
      boundaries = bucket_boundaries
    logging.debug('bucket_boundaries', boundaries)
    # with tf.compat.v1.name_scope("bucket_by_seq_length"):
    def example_to_bucket_id(*args, **kw):
      """Return int64 id of the length bucket for this example."""
      #assert length_index is not None
      if length_key is None:
        try:
          x = list(args[0])[length_index]
        except Exception:
          x = args[length_index]
      else:
        try:
          x = args[0][length_key]
        except Exception:
          x = args[length_key]      
      
      # seq_length = tf.reduce_sum(input_tensor=tf.cast(tf.cast(x, tf.bool), tf.int32))
      seq_length = tf.size(x)
      
      buckets_min = [np.iinfo(np.int32).min] + boundaries
      buckets_max = boundaries + [np.iinfo(np.int32).max]
      conditions_c = tf.logical_and(
          tf.less_equal(buckets_min, seq_length),
          tf.less(seq_length, buckets_max))
      bucket_id = tf.reduce_min(input_tensor=tf.where(conditions_c))
      return bucket_id

    if not bucket_batch_sizes:
      def batching_fn(bucket_id, grouped_d):
          return grouped_d.padded_batch(batch_size, padded_shapes=shapes, padding_values=padding_values)

      ## TODO larger window better hsku squad doing this like below, shuffle can be better ?
      ## NOTICE!! shuffle may be slow start fill queue can remove not hurt performance ?
      d = d.apply(tf.data.experimental.group_by_window(
        example_to_bucket_id, batching_fn, window_size=5 * batch_size)).shuffle((len(boundaries) + 1) * 25, seed=seed)

      ## tenor2tensor doing this, no shuffle ? also it seems like window_func for different bounds
      ## with different batch_size ?
      # d = d.apply(
      #   tf.contrib.data.group_by_window(example_to_bucket_id, batching_fn, batch_size)).shuffle((len(boundaries) + 1) * 25)
    else:
      # TEST OK 
      # test ok ie buckets[400] batch_sizes[64, 32]
      if not isinstance(bucket_batch_sizes, (list, tuple)):
        bucket_batch_sizes = [int(x) for x in bucket_batch_sizes.split(',') if x.strip()]

      logging.debug('bucket_batche_sizes', bucket_batch_sizes)
      assert len(boundaries) + 1 == len(bucket_batch_sizes)

      def window_size_fn(bucket_id):
        # window size = batch size
        batch_sizes = tf.constant(bucket_batch_sizes, dtype=tf.int64)
        window_size = batch_sizes[bucket_id]
        # * 5 will make reading slower
        window_size *= 5
        return window_size

      def batching_fn(bucket_id, grouped_d):
        batch_sizes = tf.constant(bucket_batch_sizes, dtype=tf.int64)
        batch_size = batch_sizes[bucket_id]
        #return padded_batch(grouped_d, batch_size, padded_shapes=None)
        return grouped_d.padded_batch(batch_size, padded_shapes=(shapes), padding_values=padding_values)

      # shuffle will make start slower might fill
      d = d.apply(tf.data.experimental.group_by_window(
        example_to_bucket_id, batching_fn, None, window_size_fn)).shuffle((len(boundaries) + 1) * 25, seed=seed)      
  else:
    # ----------------- no bucket
    # 注意如果有varlen 并且 不是batch_parse模式 必须要dynamic_pad 走padded_batch 因为单个组batch有不同长度 必须得padded_batch 前面有sparse2dense也没用因为是单个sparse2dense 
    # 只是sparse转成dense了 但是长度还是不能对齐
    # assert batch_parse or dynamic_pad, 'must set dynamic_pad=True to use padded_batch with parse_single_example for batch_parse=0'
    # ic(shapes)
    if dynamic_pad:
      if not batch_parse: # 似乎batch_parse=0 也就是parse_single_example必须依赖 padded_batch 而 batch_parse模式不依赖
        if tf.__version__ >= '2.2':
          # 2.2 版本开始可以不输入padded_shapes 
          d = d.padded_batch(batch_size, drop_remainder=drop_remainder)
        else:
          d = d.padded_batch(batch_size, padded_shapes=(shapes), padding_values=padding_values, drop_remainder=drop_remainder)
      else:
        d = d.batch(batch_size, drop_remainder=drop_remainder)
    else:
      d = d.batch(batch_size, drop_remainder=drop_remainder)
      
    if batch_parse:
      d = d.map(decode_fn, num_parallel_calls=AUTO)
      if filter_fn is not None:
        try:
          d = d.unbatch()
          d = d.filter(filter_fn)
        except Exception:
          d = d.unbatch()
          # TODO seems only ok in tf2 ?
          # example https://www.kaggle.com/kivlichangoogle/jigsaw-multilingual-getting-started
        
        d = d.batch(batch_size, drop_remainder=drop_remainder)
  
    if post_decode_fn is not None:
      d = d.map(post_decode_fn, num_parallel_calls=AUTO)

  if cache:
    logging.debug('Cache datase after map')
    d = d.cache(cache_file)
    
  d = d.prefetch(FLAGS.prefetch or AUTO)
  # if not device or tf.executing_eagerly():
  #   d = d.prefetch(buffer_size=AUTO)
  #   # d = d.prefetch(buffer_size=51200)
  # else:
  #   d = d.apply(tf.data.experimental.prefetch_to_device(device))
    
  if not FLAGS.keras and tf.__version__ > '2':
    if melt.distributed.has_strategy(distribute_strategy):
      strategy = distribute_strategy or melt.distributed.get_strategy()
      d = strategy.experimental_distribute_dataset(d)
      return d

  # if not allow_smaller_final_batch:
  #   # https://github.com/tensorflow/tensorflow/issues/13745 d.apply(tf.contrib.data.batch_and_drop_remainder(10)).
  #   d = d.filter(lambda x, *args, **kw: tf.equal(tf.shape(x)[0], batch_size))

  # TODO save iterator ?
  ## Create saveable object from iterator.
  #saveable = tf.contrib.data.make_saveable_from_iterator(iterator)

  # Save the iterator state by adding it to the saveable objects collection.
  #tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable)
    #try:
  if tf.executing_eagerly() or tf.__version__ >= '2':
    # TODO store iterator for eager, 注意 as_numpy_iterator就只能访问一次了。。
    if not return_numpy:    
      return d
    else:
      return d.as_numpy_iterator()
  else:
    #if repeat and not initializable:
    if not initializable:
      iterator = tf.compat.v1.data.make_one_shot_iterator(d) 
      # saveable = tf.data.experimental.make_saveable_from_iterator(iterator)
      # tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.SAVEABLE_OBJECTS, saveable)
      if return_iterator:
        return iterator
      ops = iterator.get_next()
      return ops
    else:
      iterator = tf.compat.v1.data.make_initializable_iterator(d)
      # saveable = tf.data.experimental.make_saveable_from_iterator(iterator)
      # tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.SAVEABLE_OBJECTS, saveable)
      return iterator
  # except Exception:
  #   if repeat and not initializable:
  #     iterator = d.make_one_shot_iterator()
  #     saveable = tf.contrib.data.make_saveable_from_iterator(iterator)
  #     tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable)
  #     if return_iterator:
  #       return iterator
  #     ops = iterator.get_next()
  #     return ops
  #   else:
  #     # if not repeat then need to init iterator each epoch
  #     iterator = d.make_initializable_iterator()
  #     saveable = tf.contrib.data.make_saveable_from_iterator(iterator)
  #     tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable)
  #     return iterator         
