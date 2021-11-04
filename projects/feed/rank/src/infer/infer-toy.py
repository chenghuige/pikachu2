#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   get-all-uers.py
#        \author   chenghuige  
#          \date   2019-08-18 11:06:39.496266
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import collections
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import random
import traceback
import pandas as pd
import glob
import six

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from absl import app, flags
FLAGS = flags.FLAGS

def glob_(x):
  if x.startswith('gs://'):
    return tf.io.gfile.glob(x)
  else:
    return glob.glob(x)

def exists_(x):
  if x.startswith('gs://'):
    return tf.io.gfile.exists(x)
  else:
    return os.path.exists(x)

def is_glob_pattern(input):
  return '*' in input

def file_is_empty(path):
  if path.startswith('gs://'):
    return False
  # HACK for CloudS file
  if 'CloudS' in os.path.realpath(path) and os.path.basename(path).startswith('tfrecord'):
    return False
  try:
    return os.stat(path).st_size==0
  except Exception:
    return True

def list_files(inputs):
  if not inputs:
    return []
  files = []
  inputs = inputs.split(',')
  for input in inputs:
    if not input or not input.strip():
      continue
    parts = []
    if tf.io.gfile.isdir(input):
      parts = glob_(f'{input}/*')
    else:
      parts = glob_(input)
    files += parts 

  def _is_bad_name(x):
    return x.endswith('num_records.txt') \
              or x.endswith('.idx') or x.startswith('.')  \
              or x.startswith('_') or 'COPYING' in x or 'TMP' in x or 'TEMP' in x

  files = [x for x in files if exists_(x) \
            and (x.startswith('gs://') or os.path.isfile(x)) \
            and not file_is_empty(x) \
            and not _is_bad_name(os.path.basename(x))]
  return files

def get_num_records_single(tf_record_file):
  num_records = 0
  filename = os.path.basename(tf_record_file)
  l = filename.split('.')
  try:
    num_records = int(l[-1])
  except Exception:
    pass 
  
  if num_records > 10:
    return num_records
  
  try:
    return sum(1 for _ in tf.compat.v1.python_io.tf_record_iterator(tf_record_file))
  except Exception:
    return 0

def get_num_records(files):
  if isinstance(files, str):
    files = gezi.list_files(files) 
  # print(files, file=sys.stderr)
  return sum([get_num_records_single(file) for file in tqdm(files, ascii=True, desc='get_num_records', leave=False)])
  # return sum([get_num_records_single(file) for file in files])

def squeeze_(x, dim=-1):
  if len(x.shape) > 1 and x.shape[dim] == 1:
    return x.squeeze(dim)
  return x

def decode_example(x):
  if tf.executing_eagerly():
    x = x.numpy()
  x = tf.train.Example.FromString(x).features.feature
  features = {}
  for key in x.keys():
    typenames = ['bytes_list', 'float_list', 'int64_list']
    dtypes = [np.object, np.float32, np.int64]
    for typename, dtype in zip(typenames, dtypes):
      value = getattr(x[key], typename).value
      if value:
        features[key] = np.array(value, dtype=dtype)
  return features

def first_example(record_file):
  if isinstance(record_file, (list, tuple)):
    record_file = record_file[0]
  if tf.executing_eagerly():
    for item in tf.data.TFRecordDataset(record_file):
      x = decode_example(item)
      return x
  else:
    for item in tf.compat.v1.python_io.tf_record_iterator(record_file):
      x = decode_example(item)
      return x

def sparse2dense(features, key=None, default_value=0):
  def sparse2dense_(features, key, default_value):
    val = features[key]
    val = tf.sparse.to_dense(val, default_value)
    features[key] = val   
  if key:
    sparse2dense(features, key)
  else:
    from tensorflow.python.framework.sparse_tensor import SparseTensor
    for key, val in features.items():
      if isinstance(val, SparseTensor):
        if val.values.dtype == tf.string:
          default_value = '\x01'
        sparse2dense_(features, key, default_value)

def decode(bytes_list):
  if not six.PY2:
    if bytes_list.dtype in [int, float, np.int32, np.int64, np.float32]:
      return bytes_list
    import tensorflow as tf
    try:
      return np.array([tf.compat.as_str_any(x) for x in bytes_list])
    except Exception:
      return bytes_list
  else:
    return bytes_list

class Dataset():
  def __init__(self):
    self.batch_parse = True
    self.parse_fn = tf.io.parse_single_example if not self.batch_parse else tf.io.parse_example

  def gen_example(self, files=None):
    if not files:
      files = self.get_filenames()
    if not isinstance(files, (list, tuple)):
      files = [files]
    example = {}
    if files:
      for file in files:
        example = first_example(file)
        if example:
          self.example = example
          break
    self.example = example
    assert self.example, files
    return example

  def adds(self, features_dict, names, dtype=None, length=None):
    dtype_ = dtype
    for name in names:
      if name in self.example:
        dtype = dtype_ or self.example[name].dtype 
        if length is None:
          features_dict[name] = tf.io.VarLenFeature(dtype)
        elif length > 0:
          features_dict[name] = tf.io.FixedLenFeature([length], dtype)
        else:
          features_dict[name] = tf.io.FixedLenFeature([], dtype)
    
  def parse_onehot(self):
    features_dict = {
      'index': tf.io.VarLenFeature(tf.int64),
      'field': tf.io.VarLenFeature(tf.int64),
      'value': tf.io.VarLenFeature(tf.float32)
    }
    return features_dict
    
  def parse(self, example):
    features_dict = {
      'id':  tf.io.FixedLenFeature([], tf.string),
      'mid': tf.io.FixedLenFeature([], tf.string),
      'docid': tf.io.FixedLenFeature([], tf.string),
      # 'click': tf.io.FixedLenFeature([], tf.int64),
      'duration': tf.io.FixedLenFeature([], tf.int64),
      'uid': tf.io.FixedLenFeature([1], tf.int64),
      'did': tf.io.FixedLenFeature([1], tf.int64)
      }
    
    # if FLAGS.compare_online:
    features_dict_ = {
      'abtestid': tf.io.FixedLenFeature([], tf.int64),
      'show_time': tf.io.FixedLenFeature([], tf.int64),
      'ori_lr_score': tf.io.FixedLenFeature([], tf.float32),
      'lr_score': tf.io.FixedLenFeature([], tf.float32),
      'position': tf.io.FixedLenFeature([], tf.int64),
      'video_time': tf.io.FixedLenFeature([], tf.int64),
      'impression_time': tf.io.FixedLenFeature([], tf.int64),
      'article_page_time': tf.io.FixedLenFeature([], tf.int64),
      'rea': tf.io.FixedLenFeature([1], tf.string),
      'type': tf.io.FixedLenFeature([1], tf.int64),
      'time_interval': tf.io.FixedLenFeature([1], tf.int64),
      'time_weekday': tf.io.FixedLenFeature([1], tf.int64),
      'timespan_interval': tf.io.FixedLenFeature([1], tf.int64),
    }

    def _adds(names, dtype=None, length=None):
      self.adds(features_dict, names, dtype, length)

    inames = [
      'user_active', 'mobile_screen_width', 'type',
      'network', 'today_refresh_num', 'coldstart_refresh_num',
      # 'distribution_id', 
      'mktest_distribution_id_feed',
    ]
    fnames = ['read_completion_rate', 'ol_pred', 'ol_pred_click', 'ol_pred_dur']
    snames = ['product', 'distribution', 'model_name', 'model_ver', 'mobile_brand', 'mobile_model']

    _adds(inames, tf.int64, 1)
    _adds(fnames, tf.float32, 1)
    _adds(snames, tf.string, 1)    
    
    features_dict.update(features_dict_)

    features_dict_ = {
      'keyword': tf.io.VarLenFeature(tf.int64),
      'topic': tf.io.VarLenFeature(tf.int64),
      'history': tf.io.VarLenFeature(tf.int64),
      'doc_keyword': tf.io.VarLenFeature(tf.int64),
      'doc_topic': tf.io.VarLenFeature(tf.int64),
      'tw_history': tf.io.VarLenFeature(tf.int64),  # add
      'tw_history_topic': tf.io.VarLenFeature(tf.int64),  # add
      'tw_history_rec': tf.io.VarLenFeature(tf.int64),  # add
      'tw_history_kw': tf.io.VarLenFeature(tf.int64),  # add
      'vd_history': tf.io.VarLenFeature(tf.int64),  # add
      'vd_history_topic': tf.io.VarLenFeature(tf.int64),  # add
      'mktest_tw_history_kw_feed': tf.io.VarLenFeature(tf.int64),  # mkyuwen 0520
      'mktest_vd_history_kw_feed': tf.io.VarLenFeature(tf.int64),  # mkyuwen 0521
      'mktest_rel_vd_history_kw_feed': tf.io.VarLenFeature(tf.int64),
      'mktest_doc_kw_feed': tf.io.VarLenFeature(tf.int64),
      'mktest_doc_kw_secondary_feed': tf.io.VarLenFeature(tf.int64),
      'mktest_tw_long_term_kw_feed': tf.io.VarLenFeature(tf.int64),
      'mktest_vd_long_term_kw_feed': tf.io.VarLenFeature(tf.int64),
      'mktest_new_search_kw_feed': tf.io.VarLenFeature(tf.int64),
      'mktest_long_search_kw_feed': tf.io.VarLenFeature(tf.int64),
      'mktest_user_kw_feed': tf.io.VarLenFeature(tf.int64),

      #'mktest_new_search_kw_score_feed': tf.io.VarLenFeature(tf.float32), # 0522
      #'mktest_user_kw_score_feed': tf.io.VarLenFeature(tf.float32),  # 0522
      #'mktest_doc_kw_score_feed': tf.io.VarLenFeature(tf.float32), # 0522
      #'mktest_doc_kw_secondary_score_feed': tf.io.VarLenFeature(tf.float32), # 0522
      }
    features_dict_.update(self.parse_onehot())

    features_dict.update(features_dict_)

    features_dict['num_interests'] = tf.io.FixedLenFeature([], tf.int64)
    features_dict['unlike'] = tf.io.FixedLenFeature([], tf.int64)
  
    features = self.parse_fn(serialized=example, features=features_dict)
    sparse2dense(features, default_value=0)
    features['click'] = tf.cast(K.not_equal(features['duration'], 0), tf.int64)

    if 'read_completion_rate' not in features:
      features['read_completion_rate'] = tf.zeros_like(features['duration'], dtype=tf.float32)
          
    #------notice features always has weight but you can choose not to use it by setting FLAGS.use_weight=0 by default will be True
    features['weight'] = 1.
    def _append_dim(features, keys):
      for key in keys:
        features[key] = tf.expand_dims(features[key], axis=-1)
    
    _append_dim(features, ['click', 'duration', 'weight'])

    y = features['click']
    y = tf.cast(y, tf.float32) 

    x = features
    return x, y

  def make_batch(self, batch_size, files):
    d = tf.data.TFRecordDataset(files)
    AUTO = tf.data.experimental.AUTOTUNE
    d = d.batch(batch_size).map(self.parse, num_parallel_calls=AUTO).prefetch(AUTO)
    return  tf.compat.v1.data.make_one_shot_iterator(d)

def get_model_dir_and_path(model_dir, model_name=None):
  model_path = model_dir
  ckpt = tf.train.get_checkpoint_state(model_dir)
  if ckpt and ckpt.model_checkpoint_path:
    #model_path = '%s/%s'%(model_dir, os.path.basename(ckpt.model_checkpoint_path)) 
    model_path = os.path.join(model_dir, os.path.basename(ckpt.model_checkpoint_path))
  else:
    model_path = model_dir if model_name is None else os.path.join(model_dir, model_name)
  #if not os.path.exists(model_path):
  #  raise ValueError(model_path)
  return os.path.dirname(model_path), model_path

def get_tensor_from_key(key, graph, index=-1):
  if isinstance(key, str):
    if not '/' in key:
      try:
        ops = graph.get_collection(key)
        if len(ops) > 1:
          #print('Warning: ops more then 1 for {}, ops:{}, index:{}'.format(key, ops, index))
          pass
        return ops[index]
      except Exception:
        pass
    else:
      if not key.endswith(':0'):
        key = key + ':0'
      #print('------------', [n.name for n in graph.as_graph_def().node])
      try:
        op = graph.get_tensor_by_name(key)
        return op
      except Exception:
        #print(traceback.format_exc())
        # key = 'prefix/' + key 
        op = graph.get_tensor_by_name(key)
        return op
  else:
    return key

class Predictor(object):
  def __init__(self, model_dir=None, meta_graph=None, model_name=None, 
               debug=False, sess=None, graph=None,
               frozen_graph=None, frozen_graph_name='',
               random_seed=1234):
    super(Predictor, self).__init__()
    self.sess = sess
    if self.sess is None:
      ##---TODO tf.Session() if sess is None
      #self.sess = tf.InteractiveSession()
      #self.sess = melt.get_session() #make sure use one same global/share sess in your graph
      self.graph = graph or tf.Graph()
      self.sess = melt.get_session(graph=self.graph) #by default to use new Session, so not conflict with previous Predictors(like overwide values)
      if debug:
        self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
    else:
      self.graph = self.sess.graph
    #ops will be map and internal list like
    #{ text : [op1, op2], text2 : [op3, op4, op5], text3 : [op6] }
    
    if random_seed is not None:
      tf.compat.v1.set_random_seed(random_seed)

    self.frozen_graph_name = frozen_graph_name
    if frozen_graph is None:
      if model_dir is not None and os.path.isdir(model_dir):
        self.restore(model_dir, meta_graph, model_name)
      elif os.path.isfile(model_dir):
       	self.load_graph(model_dir, frozen_graph_name)
      else:
        logging.warning(f'{model_dir} not ok')
        raise ValueError(model_dir) 
    else:
      self.load_graph(frozen_graph, frozen_graph_name)

  #by default will use last one
  def inference(self, key, feed_dict=None, index=-1, return_dict=False, **kwargs):
    if not isinstance(key, (list, tuple)):
      return self.sess.run(get_tensor_from_key(key, self.graph, index), feed_dict=feed_dict)
    else:
      keys = key 
      if not isinstance(index, (list, tuple)):
        indexes = [index] * len(keys)
      else:
        indexes = index 
      tensors = [get_tensor_from_key(key, self.graph, index) for key,index in zip(keys, indexes)]
      if not return_dict:
        return self.sess.run(tensors, feed_dict=feed_dict, **kwargs)
      else:
        m = dict(zip(keys, tensors))
        m2 = {}
        for key in m:
          if m[key] is not None:
            m2[key] = m[key]
        return self.sess.run(m2, feed_dict=feed_dict, **kwargs)

  def predict(self, key, feed_dict=None, index=-1, return_dict=False):
    return self.inference(key, feed_dict, index, return_dict)

  def run(self, key, feed_dict=None):
    return self.sess.run(key, feed_dict)

  def restore(self, model_dir, meta_graph=None, model_name=None, random_seed=None):
    """
    do not need to create graph
    restore graph from meta file then restore values from checkpoint file
    """
    model_dir, model_path = get_model_dir_and_path(model_dir, model_name)
    self.model_path = model_path

    frozen_graph_file = '%s.pb' % model_path
    if os.path.exists(frozen_graph_file):
      print('Loading from frozen_graph', frozen_graph_file, file=sys.stderr)
      frozen_map_file = '%s.map' % model_path
      return self.load_graph(frozen_graph_file, self.frozen_graph_name, frozen_map_file=frozen_map_file)

    if meta_graph is None:
      meta_graph = '%s.meta' % model_path
    assert os.path.exists(meta_graph), 'no pb and meta_graph: %s' % model_path
    ##https://github.com/tensorflow/tensorflow/issues/4603
    #https://stackoverflow.com/questions/37649060/tensorflow-restoring-a-graph-and-model-then-running-evaluation-on-a-single-imag
    with self.sess.graph.as_default():
      timer = gezi.Timer(f'Restoring {model_path}', print_fn=logging.info)
      saver = tf.compat.v1.train.import_meta_graph(meta_graph)
      saver.restore(self.sess, model_path)
      timer.print()
      try:
        self.sess.run(tf.compat.v1.tables_initializer())
      except Exception:
        pass

    if random_seed is not None:
      tf.compat.v1.set_random_seed(random_seed)

    #---so maybe do not use num_epochs or not save num_epochs variable!!!! can set but input producer not use, stop by my flow loop
    #---TODO not work remove can run but hang  FIXME add predictor + exact_predictor during train will face
    #@gauravsindhwani , can you still run the code successfully after you remove these two collections since they are actually part of the graph. 
    #I try your way but find the program is stuck after restoring."
    #https://github.com/tensorflow/tensorflow/issues/9747
    #tf.get_default_graph().clear_collection("queue_runners")
    #tf.get_default_graph().clear_collection("local_variables")
    #--for num_epochs not 0
    #tf.get_default_graph().clear_collection("local_variables")
    #self.sess.run(tf.local_variables_initializer())

    #https://stackoverflow.com/questions/44251666/how-to-initialize-tensorflow-variable-that-wasnt-saved-other-than-with-tf-globa
    #melt.initialize_uninitialized_vars(self.sess)

    return self.sess

  #http://ndres.me/post/convert-caffe-to-tensorflow/
  def load_graph(self, frozen_graph_file, frozen_graph_name='', frozen_map_file=None):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    if not frozen_map_file:
      frozen_map_file = frozen_graph_file.replace('.pb', '.map')
    print('load frozen graph from %s with mapfile %s' % (frozen_graph_file, frozen_map_file))
    with tf.io.gfile.GFile(frozen_graph_file, "rb") as f:
      graph_def = tf.compat.v1.GraphDef()
      graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with self.sess.graph.as_default() as graph:
      # tf.train.import_meta_graph(frozen_graph_file)
      tf.import_graph_def(
        graph_def,
        input_map=None,
        return_elements=None,
        name=frozen_graph_name,
        #op_dict=None,
        producer_op_list=None
      )
      try:
      	self.sess.run(graph.get_operation_by_name('init_all_tables'))
      except KeyError:
        pass
      if frozen_map_file is not None and os.path.exists(frozen_map_file):
        for line in open(frozen_map_file):
          cname, key = line.strip().split('\t')
          if 'initializer' in cname:
          	continue
          if not (key.endswith(':0') or key.endswith(':1') or key.endswith(':2')):
            key = '%s:0' % key
          tensor_name = '%s/%s' % (frozen_graph_name, key) if frozen_graph_name else key
          tensor = graph.get_tensor_by_name(tensor_name)
          graph.add_to_collection(cname, tensor)

    return graph

def main(_):      
  in_dir = os.path.realpath(sys.argv[1])
  hour = os.path.basename(in_dir)
  files = list_files(in_dir)
  num_records_file = os.path.join(in_dir, "num_records.txt")
  total = get_num_records(files) 
  print('total', total, in_dir,  file=sys.stderr)

  if not total:
    exit(1)

  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  with sess.graph.as_default():
    model_path = os.path.realpath(sys.argv[2])
    # predictor = Predictor(model_path, sess=sess)   
    predictor = Predictor(frozen_graph=f'{model_path}/model.pb', sess=sess)   
    assert predictor.graph.get_collection('index_feed') 

    dataset = Dataset()
    dataset.gen_example(files)
    res = []
    batch_size = 512
    iter = dataset.make_batch(batch_size, files)
    op = iter.get_next()

    num_steps = -int(-total // batch_size)
    print('----num_steps', num_steps, file=sys.stderr) 

    desc = 'infer' 
    m = defaultdict(list)
    for i, _ in tqdm(enumerate(range(num_steps)), total=num_steps, ascii=True, desc=desc):
      x, y = sess.run(op)
      
      if 'product' in x:
        product = x['product']
        x['product_data'] = x['product']
        del x['product']
      else:
        product = np.asarray(['sgsapp'] * len(mids))
        x['product_data'] = product
        
      x2 = {}
      bs = len(x['id'])
      keys = list(x.keys())
      for key in keys:
        if not len(x[key]):
          continue
        if key in ["tw_history","tw_history_topic","tw_history_rec","tw_history_kw","vd_history","vd_history_topic","doc_keyword","doc_topic"]:
          continue
        # mkyuwen
        if "mktest" in key and "_kw_feed" in key:
          continue
        if x[key].shape == (bs, 1):
          x[key] = squeeze_(x[key])
        if x[key].shape != (bs,):
          continue
        if x[key].dtype == np.object:
          x[key] = decode(x[key])
        x2[key] = x[key]
        m[key] += [x2[key]]

      index = x['index']
      value = x['value']
      field = x['field']
      mids = x['mid']
      docids = x['docid']
      uid = x['uid']
      did = x['did']
      history = x['history']
      product = x['product_data']
      def to_product_id(x):
        if x == 'sgsapp':
          return 0
        elif x == 'newmse':
          return 1 
        elif x == 'shida':
          return 2
        else:
          return 0
      product_ = np.asarray([to_product_id(x) for x in product])

      index, value, field = x['index'], x['value'], x['field']
      assert len(predictor.graph.get_collection('index_feed')) == 1
      feed_dict = {
                    predictor.graph.get_collection('index_feed')[-1]: index,
                    predictor.graph.get_collection('value_feed')[-1]: value,
                    predictor.graph.get_collection('field_feed')[-1]: field,
                    predictor.graph.get_collection('uid_feed')[-1]: uid.reshape(-1, 1),
                    predictor.graph.get_collection('did_feed')[-1]: did.reshape(-1, 1),
                    #predictor.graph.get_collection('doc_idx_feed')[-1]: history,
                  } 

      try:
        feed_dict.update({
                    predictor.graph.get_collection('time_interval_feed')[-1]: x['time_interval'].reshape(-1, 1),
                    predictor.graph.get_collection('time_weekday_feed')[-1]: x['time_weekday'].reshape(-1, 1),
                    predictor.graph.get_collection('timespan_interval_feed')[-1]: x['timespan_interval'].reshape(-1, 1),             
                    predictor.graph.get_collection('product_feed')[-1]: product_.reshape(-1, 1),
                    predictor.graph.get_collection('doc_kw_idx_feed')[-1]: x['doc_keyword'],
                    predictor.graph.get_collection('doc_topic_idx_feed')[-1]: x['doc_topic'].reshape(-1,1),
                    predictor.graph.get_collection('tw_history_feed')[-1]: x['tw_history'],
                    predictor.graph.get_collection('tw_history_topic_feed')[-1]: x['tw_history_topic'],
                    predictor.graph.get_collection('tw_history_rec_feed')[-1]: x['tw_history_rec'],
                    predictor.graph.get_collection('tw_history_kw_feed')[-1]: x['tw_history_kw'],
                    predictor.graph.get_collection('vd_history_feed')[-1]: x['vd_history'],
                    predictor.graph.get_collection('vd_history_topic_feed')[-1]: x['vd_history_topic'],
                    predictor.graph.get_collection('user_active_feed')[-1]: x['user_active'].reshape(-1,1),
                    predictor.graph.get_collection('rea_feed')[-1]: x['rea'].astype(int).reshape(-1,1)
          })
      except Exception:
        print("mktest infer error", traceback.format_exc(), file=sys.stderr)
        # print(traceback.format_exc(), file=sys.stderr)
        pass

      try:
        feed_dict.update({
                    predictor.graph.get_collection('mktest_distribution_id_feed')[-1]: x['mktest_distribution_id_feed'].reshape(-1, 1),

                    predictor.graph.get_collection('mktest_tw_history_kw_feed')[-1]: x['mktest_tw_history_kw_feed'],
                    predictor.graph.get_collection('mktest_vd_history_kw_feed')[-1]: x['mktest_vd_history_kw_feed'],
                    predictor.graph.get_collection('mktest_rel_vd_history_kw_feed')[-1]: x['mktest_rel_vd_history_kw_feed'],
                    predictor.graph.get_collection('mktest_tw_long_term_kw_feed')[-1]: x['mktest_tw_long_term_kw_feed'],
                    predictor.graph.get_collection('mktest_vd_long_term_kw_feed')[-1]: x['mktest_vd_long_term_kw_feed'],
                    predictor.graph.get_collection('mktest_long_search_kw_feed')[-1]: x['mktest_long_search_kw_feed'],
                    predictor.graph.get_collection('mktest_new_search_kw_feed')[-1]: x['mktest_new_search_kw_feed'],
                    predictor.graph.get_collection('mktest_user_kw_feed')[-1]: x['mktest_user_kw_feed'],

                    predictor.graph.get_collection('mktest_doc_kw_feed')[-1]: x['mktest_doc_kw_feed'],
                    predictor.graph.get_collection('mktest_doc_kw_secondary_feed')[-1]: x['mktest_doc_kw_secondary_feed'],
          })
      except Exception:
        print("mktest infer error", traceback.format_exc(), file=sys.stderr)
        pass
      
      # gezi.sprint(feed_dict)
      
      try:
        preds, preds_click, preds_dur = predictor.predict(['pred', 'pred_click', 'pred_dur'], feed_dict)
      except Exception:
        print("mktest error 3 preds not all in predictor",traceback.format_exc(), file=sys.stderr)
        preds = predictor.predict('pred', feed_dict)
        preds_click, preds_dur = preds, preds
      
      m['pred'] += [squeeze_(preds)]
      m['pred_click'] += [squeeze_(preds_click)]
      m['pred_dur'] += [squeeze_(preds_dur)]
      # print(predictor.predict(['dummy', 'index_feed', 'pred'], feed_dict, return_dict=True))

if __name__ == '__main__':
  app.run(main)
  
