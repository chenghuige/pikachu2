#!/usr/bin/env python
#encoding=utf8
# ==============================================================================
#          \file   util.py
#        \author   chenghuige
#          \date   2016-08-18 18:24:05.771671
#   \Description
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

import six

import collections
from collections import namedtuple
import numpy as np

import glob
import math

import re
import time
import subprocess
import inspect
import errno
import pickle
import copy

import contextlib
import pandas as pd
from tempfile import NamedTemporaryFile
from datetime import datetime, timedelta
from shutil import copyfile, copy2
from sklearn.preprocessing import normalize

from absl import flags

FLAGS = flags.FLAGS

import gezi

logging = gezi.logging
from io import StringIO
import tensorflow as tf


def is_cn(word):
  return '\u4e00' <= item[0] <= '\u9fa5'


def is_digit(s):
  return s.lstrip('+-').isdigit()


def break_sentence(sentence, max_sent_len, additional=5):
  """
  For example, for a sentence with 70 words, supposing the the `max_sent_len'
  is 30, break it into 3 sentences.

  :param sentence: list[str] the sentence
  :param max_sent_len:
  :return:
  """
  ret = []
  cur = 0
  length = len(sentence)
  while cur < length:
    if cur + max_sent_len + additional >= length:
      ret.append(sentence[cur:length])
      break
    ret.append(sentence[cur:min(length, cur + max_sent_len)])
    cur += max_sent_len
  return ret


def add_start_end(w, start='<S>', end='</S>'):
  return [start] + list(w) + [end]


def str2scores(l):
  if ',' in l:
    # this is list save (list of list)
    return np.array([float(x.strip()) for x in l[1:-1].split(',')])
  else:
    # this numpy save (list of numpy array)
    return np.array([float(x.strip()) for x in l[1:-1].split(' ') if x.strip()])


def get_unmodify_minutes(file_):
  file_mod_time = os.stat(file_).st_mtime
  last_time = (time.time() - file_mod_time) / 60
  return last_time


# get size by M
def get_size(x):
  return os.path.getsize(x) / 1024. / 1024.


# ## Well not work well, so fall back to use py2 bseg
# def to_simplify(sentence):
#   sentence = gezi.langconv.Converter('zh-hans').convert(sentence).replace('馀', '余')
#   return sentence

# def parse_list_str(input, sep=','):
#   return np.array([float(x.strip()) for x in input[1:-1].split(sep) if x.strip()])

# https://stackoverflow.com/questions/43146528/how-to-extract-all-the-emojis-from-text
# def extract_emojis(sentence):
#   import emoji
#   allchars = [str for str in sentence]
#   l = [c for c in allchars if c in emoji.UNICODE_EMOJI]
#   return l


def extract_emojis(content):
  import emoji
  emojis_list = map(lambda x: ''.join(x.split()), emoji.UNICODE_EMOJI.keys())
  r = re.compile('|'.join(re.escape(p) for p in emojis_list))
  return r.sub(r'', content)


def remove_emojis(sentence):
  import emoji
  allchars = [str for str in sentence]
  l = [c for c in allchars if c not in emoji.UNICODE_EMOJI]
  return ''.join(l)


def is_emoji(w):
  import emoji
  return w in emoji.UNICODE_EMOJI


def dict2namedtuple(thedict, name):
  thenametuple = namedtuple(name, [])
  for key, val in thedict.items():
    if not isinstance(key, str):
      msg = 'dict keys must be strings not {}'
      raise ValueError(msg.format(key.__class__))

    if not isinstance(val, dict):
      setattr(thenametuple, key, val)
    else:
      newname = dict2namedtuple(val, key)
      setattr(thenametuple, key, newname)

  return thenametuple


def csv(s):
  s = s.replace("\"", "\"\"")
  s = "\"" + s + "\""
  return s


def get_weights(weights):
  if isinstance(weights, str):
    weights = map(float, weights.split(','))
  total_weights = sum(weights)
  #alpha = len(weights) / total_weights
  alpha = 1 / total_weights
  weights = [x * alpha for x in weights]
  return weights


# ------- TODO move to gezi.math.
def probs_entropy(probs):
  e_x = np.asarray([-p_x * math.log(p_x, 2) for p_x in probs])
  entropy = np.sum(e_x)
  return entropy


def logits_entropy(logits):
  logits = np.asarray(logits)
  probs = logits / np.sum(logits)
  entropy = probs_entropy(probs)
  return entropy


def dist(x, y):
  return np.sqrt(np.sum((x - y)**2))


def cosine(a, b):
  from numpy import dot
  from numpy.linalg import norm
  return dot(a, b) / (norm(a) * norm(b))


# def softmax(x, axis=-1):
#     mx = np.amax(x, axis=axis, keepdims=True)
#     x_exp = np.exp(x - mx)
#     x_sum = np.sum(x_exp, axis=axis, keepdims=True)
#     res = x_exp / x_sum
#     return res


# https://stackoverflow.com/questions/3985619/how-to-calculate-a-logistic-sigmoid-function-in-python
def sigmoid(x):
  s = 1 / (1 + np.exp(-x))
  return s


def softmax(x, axis=None):
  import scipy
  return scipy.special.softmax(x, axis)


# https://stackoverflow.com/questions/40357335/numpy-how-to-get-a-max-from-an-argmax-result
def lookup_3d(a, idx):
  m, n = a.shape[:2]
  return a[np.arange(m)[:, None], np.arange(n), idx]


def lookup_nd(a, idx):
  return a[tuple(np.indices(a.shape[:-1])) + (idx,)]
  # return np.choose(idx, np.moveaxis(a, -1, 0))


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  # what about 4 channels ?  TODO
  return np.array(image.getdata()).reshape(
      (im_height, im_width, -1)).astype(np.uint8)[:, :, :3]


def dirname(input):
  if tf.io.gfile.isdir(input):
    return input
  else:
    dirname = os.path.dirname(input)
    if not dirname:
      dirname = '.'
    return dirname


def non_empty(file):
  return os.path.exists(file) and os.path.getsize(file) > 0


def empty_file(file):
  return not non_empty(file)


def normalize_list(x, norm='l2'):
  if norm.lower() == 'l1':
    return l1norm_list(x)
  elif norm.lower() == 'l2':
    return l2norm_list(x)
  else:
    raise ValueError(norm)


def l1norm_list(x):
  x = np.asarray(x)
  total = np.sum(x)
  if not total:
    total = 1e-5
  return x / total


def l2norm_list(x):
  x = np.asarray(x)
  total = np.sum(x * x)**0.5
  if not total:
    total = 1e-5
  return x / total


def merge_dicts(*dict_args):
  """
  Given any number of dicts, shallow copy and merge into a new dict,
  precedence goes to key value pairs in latter dicts.
  #https://stackoverflow.com/questions/38987/how-to-merge-two-dictionaries-in-a-single-expression
  """
  result = {}
  for dictionary in dict_args:
    result.update(dictionary)
  return result


def norm(text):
  return text.strip().lower().replace('。', '')


def loggest_match(cns,
                  vocab,
                  encode_unk=False,
                  unk_vocab_size=None,
                  vocab_size=None):
  len_ = len(cns)
  for i in range(len_):
    w = ''.join(cns[:len_ - i])
    #for compat with c++ vocabulary
    if vocab.has(w):
      return vocab.id(w), cns[len_ - i:]
    elif unk_vocab_size:
      return gezi.hash(w) % unk_vocab_size + vocab_size, cns[len_ - i:]
  if encode_unk:
    return vocab.unk_id(), cns[1:]
  else:
    return -1, cns[1:]


#TODO might use trie to speed up longest match segment
def loggest_match_seg(word, vocab, encode_unk=False):
  cns = gezi.get_single_cns(word)
  word_ids = []
  while True:
    id, cns = loggest_match(cns, vocab, encode_unk=encode_unk)
    if id != -1:
      word_ids.append(id)
    if not cns:
      break
  return word_ids


def index(l, val):
  try:
    return l.index(val)
  except Exception:
    return len(l)


def to_pascal_name(name):
  if not name or name[0].isupper():
    return name
  return gnu2pascal(name)


def to_gnu_name(name):
  if not name or name[0].islower():
    return name
  return pascal2gnu(name)


def pascal2gnu(name):
  """
  convert from AbcDef to abc_def
  name must be in pascal format
  """
  l = [name[0].lower()]
  for i in range(1, len(name)):
    if name[i].isupper():
      l.append('_')
      l.append(name[i].lower())
    else:
      l.append(name[i])
  return ''.join(l)


def gnu2pascal(name):
  """
  convert from abc_def to AbcDef
  name must be in gnu format
  """
  l = []
  l = [name[0].upper()]
  need_upper = False
  for i in range(1, len(name)):
    if name[i] == '_':
      need_upper = True
      continue
    else:
      if need_upper:
        l.append(name[i].upper())
      else:
        l.append(name[i])
      need_upper = False
  return ''.join(l)

  return ''.join(l)


def is_gbk_luanma(text):
  return len(text) > len(text.decode('gb2312', 'ignore').encode('utf8'))


def gen_sum_list(l):
  l2 = [x for x in l]
  for i in range(1, len(l)):
    l2[i] += l2[i - 1]
  return l2


def add_one(d, word):
  if not word in d:
    d[word] = 1
  else:
    d[word] += 1


def pretty_floats(values):
  if not isinstance(values, (list, tuple)):
    values = [values]
  return [float('{:.4f}'.format(x)) for x in values]
  #return [float('{:e}'.format(x)) for x in values]
  #return ['{}'.format(x) for x in values]


def get_singles(l):
  """
  get signle elment as list, filter list
  """
  return [x for x in l if not isinstance(x, collections.Iterable)]


def is_single(item):
  return not isinstance(item, collections.Iterable)


def iterable(item):
  """
  be careful!  like string 'abc' is iterable! 
  you may need to use if not isinstance(values, (list, tuple)):
  """
  return isinstance(item, collections.Iterable)


def is_list_or_tuple(item):
  return isinstance(item, (list, tuple))


def get_value_name_list(values, names):
  return [
      '{}:{:.4f}'.format(x[0], x[1])
      if x[0] != 'loss' else '{:.4f}'.format(x[1]) for x in zip(names, values)
  ]


def _batches(l, batch_size):
  """
  :param l:           list
  :param group_size:  size of each group
  :return:            Yields successive group-sized lists from l.
  """
  for i in range(0, len(l), batch_size):
    yield l[i:i + batch_size]


def batches(l, batch_size, worker=0, num_workers=1):
  """
  :param l:           list
  :param group_size:  size of each group
  :return:            Yields successive group-sized lists from l.
  """
  ints_per_worker = -(-len(l) // num_workers)
  start_idx = worker * insts_per_worker
  end_idx = start_idx + insts_per_worker
  end_idx = min(end_idx, len(l))
  return _batches(l[start_idx:end_idx], batch_size)


def make_batch(input, batch_size, worker=0, num_workers=1, parse_fn=None):
  batches = []
  for i, item in enumerate(input):
    idx = i % num_workers
    if idx == worker:
      if parse_fn:
        item = parse_fn(item)
      batches.append(item)
      if len(batches) == batch_size:
        yield batches
        batches = []
  if batches:
    yield batches


#@TODO better pad
def pad(l, maxlen, mark=0, last_tokens=0):
  if isinstance(l, np.ndarray):
    l = list(l)
    return np.asarray(pad(l, maxlen, mark, last_tokens))

  if isinstance(l, list):
    if len(l) < maxlen:
      l.extend([mark] * (maxlen - len(l)))
      return l
    elif len(l) > maxlen:
      if last_tokens:
        return [*l[:maxlen - last_tokens], *l[-last_tokens:]]
      else:
        return l[:maxlen]
    else:
      return l
  else:
    raise ValueError(f'not support {l}')

def reduce_len(l, maxlen, mark=0, last_tokens=0):
  if isinstance(l, np.ndarray):
    l = list(l)
    return np.asarray(reduce_len(l, maxlen, mark, last_tokens))

  if isinstance(l, list):
    if len(l) <= maxlen:
      return l
    elif len(l) > maxlen:
      if last_tokens:
        return [*l[:maxlen - last_tokens], *l[-last_tokens:]]
      else:
        return l[:maxlen]
    else:
      return l
  else:
    raise ValueError(f'not support {l}')

def pad_batch(batch, max_len=0, mark=0):
  if not max_len:
    max_len = max([len(x) for x in batch])
  batch = np.asarray([pad(x, max_len, mark) for x in batch])
  return batch


def wrap_str(text, max_len):
  import textwrap
  return '\n'.join(textwrap.wrap(text, max_len))


def limit_str(text, max_len, last_tokens, sep='|'):
  if len(text) <= max_len:
    return text
  first_tokens = max(max_len - last_tokens - len(sep), 1)
  return text[:first_tokens] + sep + text[-last_tokens:]


def extend(l, parts):
  len_ = len(l)
  remainder = len_ % parts
  if remainder == 0:
    return l
  to_add = parts - remainder
  l2 = l[-remainder:]
  l3 = []
  while len(l3) < to_add:
    l3.extend(l2)
  l3 = l3[:to_add]
  l.extend(l3)
  return l


def nppad(l, maxlen):
  if maxlen > len(l):
    return np.lib.pad(l, (0, maxlen - len(l)), 'constant')
  else:
    return l[:maxlen]


def try_mkdir(dir):
  os.makedirs(dir, exist_ok=True)


def try_mkdir2(file_):
  os.makedirs(os.path.dirname(file_), exist_ok=True)


def try_remove(filename):
  if os.path.exists(filename):
    os.remove(filename)

def remove_dir(dir):
  #import shutil
  #shutil.rmtree(dir)
  os.system(f'sudo rm -rf {dir}')

def get_dir(path):
  if os.path.isdir(path):
    return path
  return os.path.dirname(path)


#@TODO perf?
def dedupe_list(l):
  #l_set = list(set(l))
  #l_set.sort(key = l.index)
  l_set = []
  set_ = set()
  for item in l:
    if item not in set_:
      set_.add(item)
      l_set.append(item)
  return l_set


#@TODO
def parallel_run(target, args_list, num_threads):
  record = []
  for thread_index in xrange(num_threads):
    process = multiprocessing.Process(target=target,
                                      args=args_list[thread_index])
    process.start()
    record.append(process)

  for process in record:
    process.join()


import threading


def multithreads_run(target, args_list):
  num_threads = len(args_list)
  threads = []
  for args in args_list:
    t = threading.Thread(target=target, args=args)
    threads.append(t)
  for t in threads:
    t.join()


#@TODO move to bigdata_util.py


def is_empty(x):
  return x is None or len(x) == 0


#----------------file related
def is_glob_pattern(input):
  return '*' in input


def file_is_empty(path):
  if path.startswith('gs://'):
    return False
  # HACK for CloudS file
  if 'CloudS' in os.path.realpath(path) and os.path.basename(path).startswith(
      'tfrecord'):
    return False
  try:
    return os.stat(path).st_size == 0
  except Exception:
    return True


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


def sorted_ls(path, time_descending=True):
  mtime = lambda f: os.stat(os.path.join(path, f)).st_mtime
  return list(sorted(os.listdir(path), key=mtime, reverse=time_descending))


def list_models(model_dir, time_descending=True):
  """
  list all models in model_dir
  """
  files = [
      file for file in glob.glob('%s/model.ckpt-*' % (model_dir))
      if not file.endswith('.meta')
  ]
  files.sort(key=lambda x: os.path.getmtime(x), reverse=time_descending)
  return files


#----------conf
def save_conf(con):
  file = '%s.py' % con.__name__
  out = open(file, 'w')
  for key, value in con.__dict__.items():
    if not key.startswith('__'):
      if not isinstance(value, str):
        result = '{} = {}\n'.format(key, value)
      else:
        result = '{} = \'{}\'\n'.format(key, value)
      out.write(result)


def write_to_txt(data, file):
  # Hack for hdfs write
  out = NamedTemporaryFile('w')
  out.write('{}\n'.format(data))
  out.flush()
  os.system('scp %s %s' % (out.name, file))


write_txt = write_to_txt


def append_to_txt(data, file):
  # Hack for hdfs write
  if not os.path.exists(file):
    # return write_to_txt(data + '\n', file)
    return write_to_txt(data, file)
  out = NamedTemporaryFile('w')
  lines = open(file, 'r').readlines()
  lines += ['%s\n' % data]
  out.write('{}'.format(''.join(lines)))
  out.flush()
  os.system('scp %s %s' % (out.name, file))


append_txt = append_to_txt


def read_int_from(file, default_value=None):
  try:
    return int(float(open(file).readline().strip().split()
                     [0])) if os.path.isfile(file) else default_value
  except Exception:
    return default_value


read_int = read_int_from


def read_float_from(file, default_value=None):
  try:
    return float(open(file).readline().strip().split()[0]) if os.path.isfile(
        file) else default_value
  except Exception:
    return default_value


read_float = read_float_from


def read_str_from(file, default_value=None):
  try:
    return open(file).readline().strip() if os.path.isfile(
        file) else default_value
  except Exception:
    return default_value


read_str = read_str_from
read_from = read_str_from


def img_html(img):
  return '<p><a href={0} target=_blank><img src={0} height=200></a></p>\n'.format(
      img)


def text_html(text):
  return '<p>{}</p>'.format(text)


def thtml(text):
  return text_html(text)


#@TODO support *content
def hprint(content):
  print('<p>', content, '</p>')


def imgprint(img):
  print(img_html(img))


def unison_shuffle(a, b):
  """
  http://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
  """
  assert len(a) == len(b)
  try:
    from sklearn.utils import shuffle
    a, b = shuffle(a, b, random_state=0)
    return a, b
  except Exception:
    print('sklearn not installed! use numpy but is not inplace shuffle',
          file=sys.stderr)
    import numpy
    index = numpy.random.permutation(len(a))
    return a[index], b[index]


def finalize_feature(fe, mode='w', outfile='./feature_name.txt', sep='\n'):
  #print(fe.Str('\n'), file=sys.stderr)
  #print('\n'.join(['{}:{}'.format(i, fname) for i, fname in enumerate(fe.names())]), file=sys.stderr)
  #print(fe.Length(), file=sys.stderr)
  if mode == 'w':
    fe.write_names(file=outfile, sep=sep)
  elif mode == 'a':
    fe.append_names(file=outfile, sep=sep)


def write_feature_names(names,
                        mode='a',
                        outfile='./feature_name.txt',
                        sep='\n'):
  out = open(outfile, mode)
  out.write(sep.join(names))
  out.write('\n')


def get_feature_names(file_):
  feature_names = []
  for line in open(file_):
    name = line.rstrip().split('#')[0].strip()
    if not name:
      continue
    feature_names.append(name)
  return feature_names


def read_feature_names(file_):
  feature_names = []
  for line in open(file_):
    name = line.rstrip().split('#')[0].strip()
    if not name:
      continue
    feature_names.append(name)
  return feature_names


def get_feature_names_dict(file_):
  feature_names_dict = {}
  index = 0
  for line in open(file_):
    name = line.rstrip().split('#')[0].strip()
    if not name:
      continue
    feature_names_dict[name] = index
    index += 1
  return feature_names_dict


def read_feature_names_dict(file_):
  feature_names_dict = {}
  index = 0
  for line in open(file_):
    name = line.rstrip().split('#')[0].strip()
    if not name:
      continue
    feature_names_dict[name] = index
    index += 1
  return feature_names_dict


def update_sparse_feature(feature, num_pre_features):
  features = feature.split(',')
  index_values = [x.split(':') for x in features]
  return ','.join([
      '{}:{}'.format(int(index) + num_pre_features, value)
      for index, value in index_values
  ])


def merge_sparse_feature(fe1, fe2, num_fe1):
  if not fe1:
    return update_sparse_feature(fe2, num_fe1)
  if not fe2:
    return fe1
  return ','.join([fe1, update_sparse_feature(fe2, num_fe1)])


#TODO move to other place
#http://blog.csdn.net/luo123n/article/details/9999481
def edit_distance(first, second):
  if len(first) > len(second):
    first, second = second, first
  if len(first) == 0:
    return len(second)
  if len(second) == 0:
    return len(first)
  first_length = len(first) + 1
  second_length = len(second) + 1
  distance_matrix = [range(second_length) for x in range(first_length)]
  for i in range(1, first_length):
    for j in range(1, second_length):
      deletion = distance_matrix[i - 1][j] + 1
      insertion = distance_matrix[i][j - 1] + 1
      substitution = distance_matrix[i - 1][j - 1]
      if first[i - 1] != second[j - 1]:
        substitution += 1
      distance_matrix[i][j] = min(insertion, deletion, substitution)
  return distance_matrix[first_length - 1][second_length - 1]


import json
import gezi


def save_json(obj, filename, use_timer=False):
  if use_timer:
    timer = gezi.Timer("Saving to {}...".format(filename))
  with open(filename, "w") as fh:
    json.dump(obj, fh)
  if use_timer:
    timer.print_elapsed()


def load_json(filename, use_timer=False):
  if use_timer:
    timer = gezi.Timer("Loading {}...".format(filename))
  with open(filename) as fh:
    obj = json.load(fh)
  if use_timer:
    timer.print_elapsed()
  return obj


read_json = load_json


def read_json_lines(filename, use_timer=False):
  if use_timer:
    timer = gezi.Timer("Loading {}...".format(filename))
  res = []
  for line in open(filename):
    res.append(json.loads(line.strip()))
  if use_timer:
    timer.print_elapsed()
  return res


def load_pickle(filename, use_timer=False):
  if use_timer:
    timer = gezi.Timer("Loading {}...".format(filename))
  with open(filename, 'rb') as f:
    x = pickle.load(f)
  if use_timer:
    timer.print_elapsed()
  return x


read_pickle = load_pickle


def save_pickle(obj, filename):
  with open(filename, 'wb') as f:
    pickle.dump(obj, f, protocol=-1)


write_pickle = save_pickle


def dict_to_list(x):
  return list(zip(*x.items()))


def dict2list(x):
  return list(zip(*x.items()))


def dict_to_df(x):
  x_ = {}
  for key in x:
    x_[key] = [x[key]]
  return pd.DataFrame(x_)


def dict2df(x):
  return dict_to_df(x)


def strip_suffix(s, suf):
  if s.endswith(suf):
    return s[:len(s) - len(suf)]
  return s


def log(text, array):
  """Prints a text message. And, optionally, if a Numpy array is provided it
  prints it's shape, min, and max values.
  """
  text = text.ljust(12)
  text += (
      "shape: {:20}  min: {:10.5f}  max: {:10.5f} mean: {:10.5f} unique: {:5d} {}"
      .format(str(array.shape),
              np.min(array) if array.size else "",
              np.max(array) if array.size else "",
              np.mean(array) if array.size else "",
              len(np.unique(array)) if array.size else "", array.dtype))
  print(text, file=sys.stderr)


def log_full(text, array):
  """Prints a text message. And, optionally, if a Numpy array is provided it
  prints it's shape, min, and max values.
  """
  log(text, array)
  print(np.unique(array), file=sys.stderr)


def env_has(name):
  return name in os.environ and os.environ[name] != '0'


def env_get(name):
  if name in os.environ:
    return os.environ[name]
  else:
    return None


def env_set(name, val=1):
  os.environ[name] = str(val)


def has_env(name):
  return name in os.environ and os.environ[name] != '0'


def get_env(name, val=None):
  if name in os.environ:
    val = os.environ[name]
    if val.isdigit():
      val = int(val)
  return val


def get_flags(key, val=None):
  if 'FLAGS' in global_dict:
    return global_dict['FLAGS'].get(key, val)
  else:
    return val


def set_env(name, val=1):
  os.environ[name] = str(val)


def env_val(name, default=None):
  return None if name not in os.environ else os.environ[name]


global_dict = {}


def get_global(key, val=None, env=True, flags=False):
  #优先顺序 环境变量 自定义global absl.FLAGS
  val_ = None
  if env:
    val_ = get_env(key, None)
  if val_ is None:
    val_ = global_dict.get(key, None)
  if val_ is None:
    if flags:
      val_ = get_flags(key, None)
  if val_ is None:
    val_ = val
  return val_


def get(key, val=None, env=True, flags=False):
  return get_global(key, val, env, flags)


def set_global(key, val):
  global_dict[key] = val
  return val


Set = set


def set(key, val):
  return set_global(key, val)


# TODO add_global should be append
def add_global(key, val):
  if key not in global_dict:
    global_dict[key] = [val]
  else:
    global_dict[key].append(val)


def set_global_dict(dic_key, key, val):
  if dic_key not in global_dict:
    global_dict[dic_key] = {}
  global_dict[dic_key][key] = val


def is_valid_step():
  return FLAGS.work_mode == 'train' and FLAGS.valid_interval_steps and global_step(
  ) % FLAGS.valid_interval_steps == 0


def summary_scalar(key, val):
  if is_valid_step():
    set_global_dict('summaries/scalar', key, val)


def summary_embedding(key, val):
  if is_valid_step():
    set_global_dict('summaries/embedding', key, val)


def global_step():
  return get_global('global_step', 0)


def set_global_step(step):
  set_global('global_step', step)


def use_matplotlib(backend='Agg'):
  import matplotlib
  matplotlib.use(backend)


# TODO speedup ?
def decode(bytes_list):
  if not six.PY2:
    if hasattr(bytes_list, 'dtype'):
      if bytes_list.dtype in [int, float, np.int32, np.int64, np.float32]:
        return bytes_list
    # # TODO  just use bytes.decode ?
    # import tensorflow as tf
    # try:
    #   return np.array([tf.compat.as_str_any(x) for x in bytes_list])
    # except Exception:
    #   return bytes_list
    #return bytes_list.astype('U')
    if len(bytes_list.shape) > 1:
      return np.asarray([[x[0].decode(encoding='utf-8')] for x in bytes_list])
    else:
      return np.asarray([x.decode(encoding='utf-8') for x in bytes_list])
  else:
    return bytes_list


def get_fold(total, num_folds, index):
  # index is None means all
  if index is None:
    return 0, total
  elif index < 0:
    return 0, 0
  assert num_folds
  fold_size = -(-total // num_folds)
  start = fold_size * index
  end = start + fold_size if index != num_folds - 1 else total
  return start, end


def get_df_fold(df, num_folds, index):
  total = len(df)
  start, end = gezi.get_fold(total, FLAGS.num_records, index)
  df_ = df.iloc[start:end]
  return df_


def is_fold(input, fold):
  input = strip_suffix(input, '.record')
  if fold is None:
    return False
  try:
    # part-0003
    return int(re.split('-|_', input)[1]) == fold
  except Exception:
    # 3.record
    return int(re.split('.')[0]) == fold


def to_list(item):
  if not isinstance(item, (list, tuple)):
    return [item]
  return item


def repeat(iter):
  while True:
    for x in iter:
      yield x


def dict_partial(m, keys):
  return type(m)([(key, m[key]) for key in keys if key in m])


def subdict(m, keys):
  return type(m)([(key, m[key]) for key in keys if key in m])


def dict_rename(m, x, y):
  return type(m)([(key.replace(x, y), val) for key, val in m.items()])


def dict_prefix(m, x):
  return type(m)([('%s%s' % (x, key), val) for key, val in m.items()])


def dict_suffix(m, x):
  return type(m)([('%s%s' % (key, x), val) for key, val in m.items()])


def dict_del(m, x):
  if x in m:
    del m[x]


def trans_format(x, format=None):
  if not format:
    return x
  if isinstance(x, float):
    if format.startswith('%'):
      return format % x
    else:
      return format.format(x)
  return x


def pprint_df(dframe,
              keys=None,
              print_fn=print,
              desc='',
              rename_fn=None,
              format=None):
  from tabulate import tabulate
  if format is not None and keys is None:
    keys = list(dframe.columns)
  if rename_fn is not None and keys is not None:
    for key in keys:
      dframe[rename_fn(key)] = trans_format(dframe[key], format)
    keys = [rename_fn(x) for x in keys]
  if keys is not None:
    for key in keys:
      dframe[key] = dframe[key].apply(lambda x: trans_format(x, format))
    dframe = dframe[keys]
  print_fn(f'{desc}\n',
           tabulate(dframe, headers='keys', tablefmt='psql', showindex=False))


def pprint_dict(d,
                keys=None,
                print_fn=print,
                desc='',
                rename_fn=None,
                format=None):
  from tabulate import tabulate
  if format is not None and keys is None:
    keys = d.keys()
  if not rename_fn:
    rename_fn = lambda x: x
  if keys is not None:
    d = type(d)([(rename_fn(key), trans_format(d[key], format))
                 for key in keys
                 if key in d])
  df = pd.DataFrame.from_dict([d])
  pprint_df(df, print_fn=print_fn, desc=desc)


def pprint(d, keys=None, print_fn=print, desc='', rename_fn=None, format=None):
  try:
    pprint_df(d,
              print_fn=print_fn,
              rename_fn=rename_fn,
              desc=desc,
              format=format)
  except Exception:
    pprint_dict(d,
                print_fn=print_fn,
                rename_fn=rename_fn,
                desc=desc,
                format=format)


def add_sys_path(path='..'):
  sys.path.append(os.path.join(os.getcwd(), path))


def num_lines(filename):
  return int(
      subprocess.check_output("wc -l " + filename, shell=True).split()[0])


get_num_lines = num_lines


def mapcount(filename):
  import mmap
  f = open(filename, "r+")
  buf = mmap.mmap(f.fileno(), 0)
  lines = 0
  readline = buf.readline
  while readline():
    lines += 1
  return lines


def use_mpi():
  return 'OMPI_COMM_WORLD_RANK' in os.environ


# https://stackoverflow.com/questions/9256687/using-defaultdict-with-multiprocessing TODO
from multiprocessing.managers import BaseManager, DictProxy
from collections import defaultdict


class Manager(BaseManager):

  def __init__(self, **kwargs):
    super(Manager, self).__init__(kwargs)
    self.start()


Manager.register('defaultdict', defaultdict, DictProxy)


# https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string/18425523
def self_str(x):
  callers_local_vars = inspect.currentframe().f_back.f_locals.items()
  return str([k for k, v in callers_local_vars if v is x][0])


def sprint(x, prefix='------------', file=sys.stderr):
  callers_local_vars = inspect.currentframe().f_back.f_locals.items()
  print('{}{}: {}'.format(prefix,
                          str([k for k, v in callers_local_vars if v is x][0]),
                          x),
        file=file)


# https://lorexxar.cn/2016/07/21/python-tqdm/
class DummyTqdmFile(object):
  """Dummy file-like that will write to tqdm"""
  file = None

  def __init__(self, file):
    self.file = file

  def write(self, x):
    # Avoid print() second call (useless \n)
    if len(x.rstrip()) > 0:
      tqdm.write(x, file=self.file)


@contextlib.contextmanager
def stdout_redirect_to_tqdm():
  save_stdout = sys.stdout
  try:
    sys.stdout = DummyTqdmFile(sys.stdout)
    yield save_stdout
  # Relay exceptions
  except Exception as exc:
    raise exc
  # Always restore sys.stdout if necessary
  finally:
    sys.stdout = save_stdout


# https://blog.csdn.net/dcrmg/article/details/82850457


def set_timeout(num, callback=None):
  import signal

  def wrap(func):

    def handle(
        signum, frame
    ):  # 收到信号 SIGALRM 后的回调函数，第一个参数是信号的数字，第二个参数是the interrupted stack frame.
      raise RuntimeError

    def to_do(*args, **kwargs):
      try:
        signal.signal(signal.SIGALRM, handle)  # 设置信号和回调函数
        signal.alarm(num)  # 设置 num 秒的闹钟
        r = func(*args, **kwargs)
        signal.alarm(0)  # 关闭闹钟
        return r
      except RuntimeError as e:
        if callback:
          return callback()
        return None

    return to_do

  return wrap


#-----------gpu related TODO move

#https://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
from tensorflow.python.client import device_lib


def get_num_available_gpus():
  if get_env('CUDA_VISIBLE_DEVICES') == '-1':
    return 0
  else:
    import subprocess
    try:
      n = str(subprocess.check_output(["nvidia-smi", "-L"])).count('UUID')
    except Exception:
      n = 0
    return n


def get_num_specific_gpus():
  if 'CUDA_VISIBLE_DEVICES' in os.environ:
    if os.environ['CUDA_VISIBLE_DEVICES'] == '-1':
      return 0
    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    logging.debug('CUDA_VISIBLE_DEVICES is %s' %
                  (os.environ['CUDA_VISIBLE_DEVICES']))
    return num_gpus
  else:
    return None


def use_cpu():
  if 'CUDA_VISIBLE_DEVICES' in os.environ and os.environ[
      'CUDA_VISIBLE_DEVICES'] == '-1':
    return True
  else:
    return False


def get_num_gpus():
  num_gpus = get_num_specific_gpus()
  if num_gpus is None:
    num_gpus = get_num_available_gpus()
    # num_gpus = 1
  return num_gpus


def get_specific_gpus():
  gpus = []
  if 'CUDA_VISIBLE_DEVICES' in os.environ:
    if os.environ['CUDA_VISIBLE_DEVICES'] != '-1':
      return list(map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
    else:
      return [-1]
  return gpus


@set_timeout(2, lambda: [])
def get_gpus(min_free_mem=None, max_used_mem=None, env_limit=False):
  """
  When you program startup using gpu, it might cousuming 10m so set max_used_mem > 20 is fine so setting as 0 will always filter
  """
  try:
    gpu_stats = subprocess.check_output(
        ["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
  except Exception:
    return [0]
  gpu_df = pd.read_csv(StringIO(gpu_stats.decode()),
                       names=['memory.used', 'memory.free'],
                       skiprows=1)
  # print('GPU usage:\n{}'.format(gpu_df))
  gpu_df['memory.free'] = gpu_df['memory.free'].map(
      lambda x: float(x.rstrip(' [MiB]')))
  gpu_df['memory.used'] = gpu_df['memory.used'].map(
      lambda x: float(x.rstrip(' [MiB]')))
  gpus = (-gpu_df['memory.free']).argsort(kind='mergesort')

  if env_limit and 'CUDA_VISIBLE_DEVICES' in os.environ:
    flags = list(map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
  else:
    flags = list(range(len(gpus)))

  logging.debug('gpu info:', gpu_df, flags)

  if min_free_mem is not None:
    gpus = [i for i in gpus if gpu_df['memory.free'][i] >= min_free_mem]
  if max_used_mem is not None:
    gpus = [i for i in gpus if gpu_df['memory.used'][i] <= max_used_mem]
  gpus = [i for i in gpus if i in flags]

  logging.sprint(gpus)
  return gpus


class DummyContextManager(object):

  def __enter__(self):
    pass

  def __exit__(self, *args):
    pass


## TODO compare with fcntl or wrap using fcntl ?
# https://blog.csdn.net/rubbishcan/article/details/17352261
class FileLockException(Exception):
  pass


class FileLock(object):
  """ A file locking mechanism that has context-manager support so 
        you can use it in a with statement. This should be relatively cross
        compatible as it doesn't rely on msvcrt or fcntl for the locking.
    """

  def __init__(self, file_name, timeout=60, delay=10):
    """ Prepare the file locker. Specify the file to lock and optionally
            the maximum timeout and the delay between each attempt to lock.
        """
    self.is_locked = False
    self.lockfile = os.path.join(os.getcwd(), "%s.lock" % file_name)
    self.file_name = file_name
    self.timeout = timeout
    self.delay = delay
    self.fd = None

  def acquire(self):
    """ Acquire the lock, if possible. If the lock is in use, it check again
            every `wait` seconds. It does this until it either gets the lock or
            exceeds `timeout` number of seconds, in which case it throws 
            an exception.
        """
    start_time = time.time()
    while True:
      try:
        #独占式打开文件
        self.fd = os.open(self.lockfile, os.O_CREAT | os.O_EXCL | os.O_RDWR)
        break
      except OSError as e:
        if e.errno != errno.EEXIST:
          raise
        logging.warning(f'{self.lockfile} exists will delay {self.delay}s')
        time.sleep(self.delay)
        if self.timeout:
          if (time.time() - start_time) >= self.timeout:
            # raise FileLockException("Timeout occured.")
            logging.warning(f'time out for {self.lockfile}')
            break
    self.is_locked = True

  def release(self):
    """ Get rid of the lock by deleting the lockfile. 
            When working in a `with` statement, this gets automatically 
            called at the end.
        """
    #关闭文件，删除文件
    if self.is_locked:
      if self.fd:
        os.close(self.fd)
        os.unlink(self.lockfile)
      self.is_locked = False

  def __enter__(self):
    """ Activated when used in the with statement. 
            Should automatically acquire a lock to be used in the with block.
        """
    if not self.is_locked:
      self.acquire()
    return self

  def __exit__(self, type, value, traceback):
    """ Activated at the end of the with statement.
            It automatically releases the lock if it isn't locked.
        """
    if self.is_locked:
      self.release()

  def __del__(self):
    """ Make sure that the FileLock instance doesn't leave a lockfile
            lying around.
        """
    self.release()


"""
#use as:
from filelock import FileLock
with FileLock("myfile.txt"):
    # work with the file as it is now locked
    print("Lock acquired.")
"""

from os.path import join, getsize


def getdirsize(dir, type='g'):
  size = 0
  for root, dirs, files in os.walk(dir):
    size += sum([getsize(join(root, name)) for name in files])

  if type == 'g':
    size = size / 1024 / 1024 / 1024
  if type == 'm':
    size = size / 1024 / 1024
  if type == 'k':
    size = size / 1024
  return size


# like tensorboard somoothing
def smooth(vals, weight=0.6):
  if is_empty(vals) or weight <= 0.:
    return vals
  smoothed = []
  last = vals[0]
  for point in vals:
    try:
      smoothed_val = last * weight + (1 - weight) * point
    except Exception:
      smoothed_val = point
    smoothed.append(smoothed_val)
    last = smoothed_val
  return smoothed


def guess_sep(line):
  seps = ['\t', ' ', '\a', ',']
  words = 0
  sep_ = None
  for sep in seps:
    l = line.split(sep)
    if len(l) > words:
      sep_ = sep
      words = len(l)
  return sep_


def filter_all(lists, filters):
  filt = np.asarray([1] * len(lists[0]))
  for filter_ in filters:
    filt *= np.asarray(filter_)
  filt = filt.astype(bool)
  for i in range(len(lists)):
    lists[i] = lists[i][filt]


# for notebook
# def tqdm(*args, **kwargs):
#   from tqdm.auto import tqdm
#   from tqdm import tqdm as tqdm_base
#   if hasattr(tqdm_base, '_instances'):
#     for instance in list(tqdm_base._instances):
#       tqdm_base._decr_instances(instance)
#   return tqdm_base(*args, **kwargs)
# if hasattr(__builtins__,'__IPYTHON__'):
#   from tqdm.notebook import tqdm
# else:
#   from tqdm import tqdm
## TODO remove, now hack for wechat comp
if 'PATH' in os.environ and 'tione' not in os.environ['PATH']:
  from tqdm.auto import tqdm
else:
  from tqdm import tqdm


def system(command):
  logging.debug(command)
  os.system(command)


def get_hour(start, delta):
  need_convert = False
  if isinstance(start, int):
    start = str(start)
    need_convert = True
  end = (datetime.strptime(start, '%Y%m%d%H') +
         timedelta(hours=delta)).strftime('%Y%m%d%H')
  if need_convert:
    end = int(end)
  return end


def get_day(start, delta):
  need_convert = False
  pattern = '%Y%m%d%H'
  if isinstance(start, int):
    start = str(start)
    need_convert = True
  if len(start) == len('20190105'):
    pattern = '%Y%m%d'
  end = (datetime.strptime(start, pattern) + timedelta(delta)).strftime(pattern)
  if need_convert:
    end = int(end)
  return end


class DateTime(object):

  def __init__(self, start, pattern=None, day=None):
    need_convert = False
    if isinstance(start, int):
      start = str(start)
      need_convert = True

    if pattern is None:
      pattern = '%Y%m%d%H'
      if len(start) == len('20190105'):
        pattern = '%Y%m%d'
      if '/' in start:
        pattern = '%m/%d/%Y'

    if day is None:
      if not 'H' in pattern:
        day = True
      else:
        day = False

    self.day = day
    self.pattern = pattern
    self.start = datetime.strptime(start, pattern)
    self.need_convert = need_convert

  def add_hours(self, delta=1, return_str=True):
    self.start += timedelta(hours=delta)
    if not return_str:
      return self
    end = self.start.strftime(self.pattern)
    if self.need_convert:
      end = int(end)
    return end

  def add_days(self, delta=1, return_str=True):
    self.start += timedelta(delta)
    if not return_str:
      return self
    end = self.start.strftime(self.pattern)
    if self.need_convert:
      end = int(end)
    return end

  def add(self, delta=1, day=None, return_str=True):
    if day is None:
      day = self.day
    if day:
      return self.add_days(delta, return_str)
    else:
      self.add_hours(delta, return_str)

  def timestamp(self):
    return self.start.timestamp()

  def __iadd__(self, other):
    if self.day:
      self.add_days(other)
    else:
      self.add_hours(other)
    return self

  def __isub__(self, other):
    if self.day:
      self.add_days(-other)
    else:
      self.add_hours(-other)
    return self

  def __call__(self, pattern=None):
    start = self.start.strftime(pattern or self.pattern)
    if self.need_convert:
      start = int(start)
    return start


def diff_hours(d1, d2, offset=0, pattern=None):
  if pattern is None:
    pattern = '%Y%m%d%H'
  return int(
      (datetime.strptime(str(d1), pattern) -
       datetime.strptime(str(d2), pattern)).total_seconds() / 3600) + offset


def diff_days(d1, d2, offset=0, pattern=None):
  if pattern is None and len(d1) == len('20190105'):
    pattern = '%Y%m%d'
  return int(diff_hours(d1, d2, offset=offset, pattern=pattern) / 24.)


def squeeze(x, dim=-1):
  if len(x.shape) > 1 and x.shape[dim] == 1:
    return x.squeeze(dim)
  return x


def is_cpu_only():
  return get_env('CUDA_VISIBLE_DEVICES') == '-1'


def check_cpu_only():
  assert get_env('CUDA_VISIBLE_DEVICES') == '-1', 'CUDA_VISIBLE_DEVICES=-1'


def run_parallel(*fns):
  from multiprocessing import Process
  proc = []
  for fn in fns:
    if isinstance(fn, (list, tuple)):
      p = Process(target=fn[0], args=fn[1])
    else:
      p = Process(target=fn)
    p.start()
    proc.append(p)
  for p in proc:
    p.join()


def chunks(lst, n):
  """Yield successive n-sized chunks from lst."""
  l = []
  for i in range(0, len(lst), n):
    l.append(lst[i:i + n])
  return l


class DistributedWrapper:
  """wrapper for distribtued"""

  def __init__(self):
    self.inited = False
    self.dist = None
    self.is_hvd = False
    self.comm = None

    self.world_size_ = 1
    self.local_rank_ = 0
    self.rank_ = 0

    self.init()

  def init_hvd(self):
    import mpi4py
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    self.comm = comm
    mpi4py.rc.initialize = False
    if not FLAGS.torch:
      import horovod.tensorflow as hvd
    else:
      import torch
      import horovod.torch as hvd

    hvd.init()
    assert hvd.mpi_threads_supported()
    assert hvd.size() == comm.Get_size()
    self.dist = hvd
    self.is_hvd = True
    self.inited = True

    self.rank_ = hvd.rank()
    self.local_rank_ = hvd.local_rank()
    self.world_size_ = hvd.size()

  def init(self):
    if 'OMPI_COMM_WORLD_RANK' in os.environ:
      self.init_hvd()
      return

    if FLAGS.parts and FLAGS.use_shard:
      self.rank_ = FLAGS.part
      self.local_rank_ = FLAGS.part
      self.world_size_ = FLAGS.parts

    if FLAGS.ps_strategy:
      self.rank_ = FLAGS.task_index
      self.local_rank_ = FLAGS.task_index
      # TODO now one worker one gpu only
      # self.world_size_ = int(len(FLAGS.worker_hosts.split(',')))
      return

    import torch
    import torch.distributed as dist
    world_size = get_env('WORLD_SIZE')
    if world_size and int(world_size) > 1:
      # TODO do not work, still 1 gpu with all processes..
      rank = int(get_env('LOCAL_RANK') or get_env('RANK'))
      torch.cuda.set_device(rank)
      dist.init_process_group(backend='nccl')
      self.dist = dist
      self.rank_ = self.dist.get_rank()
      self.local_rank_ = self.dist.get_rank()
      self.world_size_ = self.dist.get_world_size()

    self.inited = True

  def get_rank(self):
    return self.rank_

  def local_rank(self):
    return self.local_rank_

  def is_distributed(self):
    # TODO hack not mark tf ps as distributed..
    return self.world_size_ > 1 and not FLAGS.ps_strategy

  def rank(self):
    return self.get_rank()

  def size(self):
    return self.world_size_

  def world_size(self):
    return self.world_size_

  def get_world_size(self):
    return self.world_size_


def get_summary_writer(path=None, is_tf=False):
  summary_writer = gezi.get_global('summary_writer', None)
  if not summary_writer or path:
    summary_dir = path or FLAGS.log_dir or FLAGS.model_dir or '/tmp/melt'
    try_mkdir(summary_dir)
    # ic(summary_dir)
    summary_writer = gezi.SummaryWriter(summary_dir,
                                        set_walltime=False,
                                        is_tf=is_tf)
  return summary_writer


def write_summaries(results, step, summary=None):
  if summary is None:
    summary = get_summary_writer()
  for name, val in results.items():
    if not isinstance(val, str):
      summary.scalar(name, val, step)


def gen_metrics_df(root_dir, metric_file='metrics.csv', models=[]):
  import pymp
  from multiprocessing import Manager, cpu_count
  from datetime import datetime
  dfs = Manager().list()
  pattern = f'{root_dir}/*/{metric_file}'
  files = glob.glob(pattern)
  if models:
    models = Set(models)  #set名称被gezi.set覆盖了..
    files = [x for x in files if os.path.basename(os.path.dirname(x)) in models]
  files = sorted(files, key=lambda x: os.path.getmtime(x))
  if not files:
    return None
  ps = min(len(files), cpu_count())
  with pymp.Parallel(ps) as p:
    for i in tqdm(p.range(len(files)), desc='gen_metrics_df'):
      file = files[i]
      if not gezi.non_empty(file):
        continue
      try:
        df = pd.read_csv(file)
        if df['step'].values[0] > 1:
          df['step'] = df['step'].apply(lambda x: x - df['step'].values[0])
        df['model'] = os.path.basename(os.path.dirname(file))
        root = os.path.dirname(file)
        # df['mtime'] = pd.to_datetime(os.path.getmtime(root), unit='s')
        df['mtime'] = datetime.fromtimestamp(os.path.getctime(file))
        flag_file = os.path.join(root, 'flags.txt')
        file = flag_file if os.path.exists(flag_file) else file
        # df['ctime'] = pd.to_datetime(os.path.getctime(file), unit='s')
        df['ctime'] = datetime.fromtimestamp(os.path.getctime(file))
        if not 'step' in df.columns:
          df['step'] = [x + 1 for x in range(len(df))]
        dfs.append(df)
      except Exception:
        pass
  df = pd.concat(list(dfs))
  return df


def gen_metric_df(model_dir, metric_file='metrics.csv'):
  if not os.path.exists(f'{model_dir}/{metric_file}'):
    return None
  df = pd.read_csv(f'{model_dir}/{metric_file}')
  df['model'] = os.path.basename(model_dir)
  return df


def append_df(results, ofile):
  try:
    df = pd.read_csv(ofile)
  except Exception:
    df = pd.DataFrame()
  df = df.append(results, ignore_index=True)
  df.to_csv(ofile, index=False, float_format='%.6f')


def write_df(results, ofile, mode='w'):
  if mode == 'a':
    return append_df(results, ofile)
  df = pd.DataFrame()
  df = df.append(results, ignore_index=True)
  df.to_csv(ofile, index=False, float_format='%.6f')


class DfWriter():

  def __init__(self, log_dir=None, filename='metrics.csv'):
    if not log_dir:
      log_dir = FLAGS.log_dir
    self.metric_file = f'{log_dir}/{filename}'

  def write(self, results, mode='a'):
    results = results.copy()
    results['ntime'] = pd.datetime.now()
    if mode == 'a':
      append_df(results, self.metric_file)
    else:
      write_df(results, self.metric_file)

  def append(self, results):
    header = not os.path.exists(self.metric_file)
    results['ntime'] = pd.datetime.now()
    try:
      gezi.dict2df(results).to_csv(self.metric_file,
                                  index=False,
                                  header=header,
                                  mode='a')
    except Exception as e:
      ic(e)


def codeal(df, work_fn, num_processes=None):
  from multiprocessing import Pool, Manager, cpu_count
  m = Manager().dict()

  if not num_processes:
    num_processes = cpu_count()

  def _deal(index):
    total_df = len(df)
    start, end = get_fold(total_df, num_processes, index)
    total = end - start
    df_ = df.iloc[start:end]
    l = []
    for _, row in tqdm(df_.iterrows(), total=total):
      res = work_fn(row)
      l.append(res)
    m[index] = l

  with Pool(num_processes) as p:
    p.map(_deal, range(num_processes))

  return np.concatenate([m[i] for i in range(cpu_count())])


def read_parquet(file):
  import pyarrow.parquet as pq
  t = pq.read_table(file)
  return t.to_pandas()


def read_tsv(file, header=None, sep=None, **kwargs):
  if not header:
    header = file.replace('.tsv', '.header')
  if os.path.exists(header):
    names = open(header).readline().strip().split(sep)
  else:
    header = os.path.join(os.path.dirname(file), 'header')
    if os.path.exists(header):
      names = open(header).readline().strip().split(sep)
    else:
      names = None
  df = pd.read_csv(file, sep=sep, header=None, names=names, **kwargs)
  return df


def tsv_header(file):
  return open(file).readline().strip().split('\t')


import gc


def reduce_mem(df, verbose=0):
  start_mem = df.memory_usage().sum() / 1024**2
  for col in df.columns:
    col_type = df[col].dtypes
    if col_type != object:
      c_min = df[col].min()
      c_max = df[col].max()
      if str(col_type)[:3] == 'int':
        if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
          df[col] = df[col].astype(np.int8)
        elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
          df[col] = df[col].astype(np.int16)
        elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
          df[col] = df[col].astype(np.int32)
        elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
          df[col] = df[col].astype(np.int64)
      else:
        if c_min > np.finfo(np.float16).min and c_max < np.finfo(
            np.float16).max:
          df[col] = df[col].astype(np.float16)
        elif c_min > np.finfo(np.float32).min and c_max < np.finfo(
            np.float32).max:
          df[col] = df[col].astype(np.float32)
        else:
          df[col] = df[col].astype(np.float64)
  end_mem = df.memory_usage().sum() / 1024**2
  if verbose > 0:
    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(
        start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
  gc.collect()
  return df


import threading
import queue
import time


class AsyncWorker(threading.Thread):

  def __init__(self, dataloader, total, prefetch=10):
    threading.Thread.__init__(self)
    self.req_queue = queue.Queue()
    self.ret_queue = queue.Queue()
    self.dataloader = iter(dataloader)
    self.total = total
    self.prefetch_idx = prefetch
    for i in range(self.prefetch_idx):
      self.req_queue.put(1)

  def run(self):
    while True:
      dataset_type = self.req_queue.get(block=True)
      if dataset_type is None:
        break
      batch = next(self.dataloader)
      self.req_queue.task_done()
      self.ret_queue.put(batch)

  def get(self):
    batch = self.ret_queue.get()
    self.ret_queue.task_done()
    return batch

  def __iter__(self):
    return self.get()

  def prefetch(self):
    if self.prefetch_idx < self.total:
      self.req_queue.put(1)
      self.prefetch_idx += 1

  def stop(self):
    self.req_queue.put(None)


def format_time_delta(td_object, max_parts=2):
  """
  format time delta
  """
  seconds = np.timedelta64(td_object, 's').astype(int)
  periods = [('y', 60 * 60 * 24 * 365), ('mo', 60 * 60 * 24 * 30),
             ('d', 60 * 60 * 24), ('h', 60 * 60), ('m', 60), ('s', 1)]

  strings = []
  for period_name, period_seconds in periods:
    if seconds >= period_seconds:
      period_value, seconds = divmod(seconds, period_seconds)
      strings.append('%s%s' % (period_value, period_name))
  strings = strings[:max_parts]
  return ' '.join(strings)


def dict_add(x, y):
  if isinstance(y, dict):
    for key in y:
      if key not in x:
        x[key] = y[key]
      else:
        x[key] += y[key]
  else:
    for key in x:
      x[key] += y
  return x


def dict_div(x, y):
  if isinstance(y, dict):
    for key in y:
      if key not in x:
        x[key] = y[key]
      else:
        x[key] /= y[key]
  else:
    for key in x:
      x[key] /= y
  return x


def dict_mul(x, y):
  if isinstance(y, dict):
    for key in y:
      if key not in x:
        x[key] = y[key]
      else:
        x[key] *= y[key]
  else:
    for key in x:
      x[key] *= y
  return x


def dict_add_copy(x, y):
  return dict_div(x.copy(), y)


def dict_div_copy(x, y):
  return dict_div(x.copy(), y)


def dict_mul_copy(x, y):
  return dict_mul(x.copy(), y)


def dict_sum(x):
  for key in x:
    x[key] = np.sum(x[key])
  return x


def dict_sum_copy(x):
  return dict_sum(x.copy())


class FormatDict(dict):

  def __str__(self):
    return str({
        k: round(v, 4) if isinstance(v, float) else v for k, v in self.items()
    })


def has_tpu():
  return 'TPU_NAME' in os.environ.keys()


def imread(img_path):
  import cv2
  img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img


def read_tiff(image_path, split=True):
  import skimage.io as skimg_io
  image = skimg_io.imread(image_path)
  if not split:
    return image
  nir = image[:, :, 3]
  image = image[:, :, :3]
  return image, nir


# https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
def is_interactive():
  import __main__ as main
  return not hasattr(main, '__file__')


def in_notebook():
  try:
    from IPython import get_ipython
    if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
      return False
  except Exception as e:
    return False
  return True


def in_colab():
  try:
    import google.colab
    return True
  except:
    return False


def localtime_str():
  import time
  return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


# 不能正确生效 需要在import gezi之前执行rm ln 操作
def change_colab_localtime():
  os.system('rm /etc/localtime')
  os.system('ln -s /usr/share/zoneinfo/Asia/Shanghai /etc/localtime')
  ## 显示还是早8个小时 而!date是正确的
  # print(localtime_str())


def set_pandas(max_val=500, precision=4):
  import pandas as pd
  pd.set_option('display.max_rows', max_val)
  pd.set_option('display.max_columns', max_val)
  pd.set_option('display.max_colwidth', max_val)
  pd.set_option("display.precision", precision)


def wandb_nolog():
  import logging
  logger = logging.getLogger("wandb")
  logger.setLevel(logging.ERROR)


# https://stackoverflow.com/questions/51674222/how-to-make-json-dumps-in-python-ignore-a-non-serializable-field
def safe_serialize(obj, indent=4):
  default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
  return json.dumps(obj, default=default, indent=indent)


# https://github.com/abseil/abseil-py/blob/master/absl/testing/flagsaver.py
def restore_flags(flags_file, ignore_names=[]):
  from absl.testing import flagsaver
  with open(flags_file, 'rb') as f:
    saved_flag_values = pickle.load(f)
  # flagsaver.restore_flag_values(flag_values, FLAGS)
  ignore_names = Set(ignore_names)
  flag_values = FLAGS
  new_flag_names = list(flag_values)
  for name in new_flag_names:
    if name in ignore_names:
      continue
    saved = saved_flag_values.get(name)
    if saved is None:  #这个地方仿照 flagsaver.restore_flag_values 但是修改如果FLAGS里面有 但是restore处理的flags里面没有的key 保留按照FLAGS的值而不是删除
      # If __dict__ was not saved delete "new" flag.
      # delattr(flag_values, name)
      pass
    else:
      if flag_values[name].value != saved['_value']:
        flag_values[name].value = saved['_value']  # Ensure C++ value is set.
      flag_values[name].__dict__ = saved


def restore_global_dict(global_dict_file, ignore_names=[]):
  ignore_names = Set(ignore_names)
  gd = json.load(open(global_dict_file))
  for key in gd:
    if not key.startswith('<<non-serializable') and not key in ignore_names:
      global_dict[key] = gd[key]


def restore_configs(log_dir, ignore_names=[]):
  restore_flags(f'{log_dir}/flags.pkl', ignore_names=ignore_names)
  restore_global_dict(f'{log_dir}/global.json')


# def reduce_mem_usage(df, int_cast=True, obj_to_category=False, subset=None):
#     """
#     Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.
#     :param df: dataframe to reduce (pd.DataFrame)
#     :param int_cast: indicate if columns should be tried to be casted to int (bool)
#     :param obj_to_category: convert non-datetime related objects to category dtype (bool)
#     :param subset: subset of columns to analyse (list)
#     :return: dataset with the column dtypes adjusted (pd.DataFrame)
#     """
#     import gc
#     start_mem = df.memory_usage().sum() / 1024 ** 2;
#     gc.collect()
#     print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

#     cols = subset if subset is not None else df.columns.tolist()

#     for col in tqdm(cols):
#         col_type = df[col].dtype

#         if col_type != object and col_type.name != 'category' and 'datetime' not in col_type.name:
#             c_min = df[col].min()
#             c_max = df[col].max()

#             # test if column can be converted to an integer
#             treat_as_int = str(col_type)[:3] == 'int'
#             if int_cast and not treat_as_int:
#                 treat_as_int = check_if_integer(df[col])

#             if treat_as_int:
#                 if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
#                     df[col] = df[col].astype(np.int8)
#                 elif c_min > np.iinfo(np.uint8).min and c_max < np.iinfo(np.uint8).max:
#                     df[col] = df[col].astype(np.uint8)
#                 elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
#                     df[col] = df[col].astype(np.int16)
#                 elif c_min > np.iinfo(np.uint16).min and c_max < np.iinfo(np.uint16).max:
#                     df[col] = df[col].astype(np.uint16)
#                 elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
#                     df[col] = df[col].astype(np.int32)
#                 elif c_min > np.iinfo(np.uint32).min and c_max < np.iinfo(np.uint32).max:
#                     df[col] = df[col].astype(np.uint32)
#                 elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
#                     df[col] = df[col].astype(np.int64)
#                 elif c_min > np.iinfo(np.uint64).min and c_max < np.iinfo(np.uint64).max:
#                     df[col] = df[col].astype(np.uint64)
#             else:
#                 if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
#                     df[col] = df[col].astype(np.float16)
#                 elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
#                     df[col] = df[col].astype(np.float32)
#                 else:
#                     df[col] = df[col].astype(np.float64)
#         elif 'datetime' not in col_type.name and obj_to_category:
#             df[col] = df[col].astype('category')
#     gc.collect()
#     end_mem = df.memory_usage().sum() / 1024 ** 2
#     print('Memory usage after optimization is: {:.3f} MB'.format(end_mem))
#     print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

#     return df


def reduce_mem_usage(props, fp16=False, verbose=0):
  start_mem_usg = props.memory_usage().sum() / 1024**2
  if verbose > 0:
    print("Memory usage of properties dataframe is :", start_mem_usg, " MB")
  NAlist = []  # Keeps track of columns that have missing values filled in.
  for i, col in enumerate(props.columns):
    if props[col].dtype != object:  # Exclude strings

      # Print current column type
      if i < 5 and verbose > 1:
        print("******************************")
        print("Column: ", col)
        print("dtype before: ", props[col].dtype)

      # make variables for Int, max and min
      IsInt = False
      mx = props[col].max()
      mn = props[col].min()

      # Integer does not support NA, therefore, NA needs to be filled
      if not np.isfinite(props[col]).all():
        NAlist.append(col)
        props[col].fillna(mn - 1, inplace=True)

      # test if column can be converted to an integer
      asint = props[col].fillna(0).astype(np.int64)
      result = (props[col] - asint)
      result = result.sum()
      if result > -0.01 and result < 0.01:
        IsInt = True

      # Make Integer/unsigned Integer datatypes
      if IsInt:
        if mn >= 0:
          if mx < 255:
            props[col] = props[col].astype(np.uint8)
          elif mx < 65535:
            props[col] = props[col].astype(np.uint16)
          elif mx < 4294967295:
            props[col] = props[col].astype(np.uint32)
          else:
            props[col] = props[col].astype(np.uint64)
        else:
          if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
            props[col] = props[col].astype(np.int8)
          elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
            props[col] = props[col].astype(np.int16)
          elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
            props[col] = props[col].astype(np.int32)
          elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
            props[col] = props[col].astype(np.int64)

      # Make float datatypes 32 bit
      else:
        if not fp16:
          props[col] = props[col].astype(np.float32)
        else:
          props[col] = props[col].astype(np.float16)

      if i < 5 and verbose > 1:
        # Print new column type
        print("dtype after: ", props[col].dtype)
        print("******************************")

  # Print final result
  if verbose > 0:
    print("___MEMORY USAGE AFTER COMPLETION:___")
  mem_usg = props.memory_usage().sum() / 1024**2
  if verbose > 0:
    print("Memory usage is: ", mem_usg, " MB")
    print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")
  # return props, NAlist
  return props


def gcs_cp(file_names, gcs_dir, model_dir=None, model_name=None, verbose=1):
  if not gcs_dir.startswith('gs://'):
    gcs_dir = f'gs://{gcs_dir}'
  assert model_name or model_dir
  model_name = model_name or os.path.basename(model_dir)
  try_mkdir(model_name)
  if not isinstance(file_names, (list, tuple)):
    file_names = [file_names]

  file_paths = []
  for file_name in file_names:
    file_path = file_name if not model_dir else f'{model_dir}/{file_name}'
    if '*' in file_name:
      file_paths.extend(glob.glob(file_path))
    else:
      file_paths.append(file_path)

  file_paths = Set(file_paths)

  for file_path in file_paths:
    if not os.path.exists(file_path):
      logging.warning(f'{file_path} not exists')
    os.system(f'cp -rf {file_path} {model_name}')
    if verbose:
      file_name = os.path.basename(file_path)
      logging.info(f'{gcs_dir}/{model_name}/{file_name}')
  os.system(f'gsutil -m cp -r {model_name} {gcs_dir}')
  logging.debug(f'{gcs_dir}/{model_name}')


def split_feats(feats, feat_names, feat_lens):
  for i in range(1, len(feat_lens)):
    feat_lens[i] += feat_lens[i - 1]
  feats = np.split(feats, feat_lens, axis=-1)
  res = {}
  for i, name in enumerate(feat_names):
    res[name] = feats[i]
  return res


import hashlib


def md5sum(filename, blocksize=65536):
  hash = hashlib.md5()
  with open(filename, "rb") as f:
    for block in iter(lambda: f.read(blocksize), b""):
      hash.update(block)
  return hash.hexdigest()


def unique_list(l):
  return list(dict.fromkeys(l))


# [1] [1, 3]
def get_bin(val, bins, nan_val=None):
  if val == nan_val:
    return 0
  has_nan = int(nan_val != None)
  for i, bin in enumerate(bins):
    if val <= bin:
      return i + has_nan
  return i + 1 + has_nan


class UnionSet(object):

  def __init__(self):
    self.parent = {}

  def init(self, key):
    if key not in self.parent:
      self.parent[key] = key

  def find(self, key):
    self.init(key)
    while self.parent[key] != key:
      self.parent[key] = self.parent[self.parent[key]]
      key = self.parent[key]
    return key

  def join(self, key1, key2):
    p1 = self.find(key1)
    p2 = self.find(key2)
    if p1 != p2:
      self.parent[p2] = p1

  def clusters(self):
    res = {}
    for key in self.parent:
      root = self.find(key)
      if root not in res:
        res[root] = [key]
      else:
        res[root].append(key)
    return res

def words_mask(words, tokens=None):
#   if tokens:
#     words_ = []
#     i, j = 0, 0
#     while i < len(words):
#       print(i, j)
#       if len(words[i]) < len(tokens[j]):
#         k = i + 1
#         while k < len(words):
#           word = ''.join(words[i:k])
#           print(i, len(word), len(tokens[j]), word, tokens[j])
#           if len(word) == len(tokens[j]):
#             words_.append(word)
#             break
#           else:
#             k += 1
#         i = k
#         j += 1
#       else:
#         words_.append(words[i])
#         j += len(words[i]) - len(tokens[j]) + 1
#         i += 1
#       print(words_)
#     print(words_)
#     words = words_
  
  try:
    word_lens = [len(word) for word in words]
    total_len = sum(word_lens) if not tokens else len(tokens)
    #   print(total_len)
    start = 0
    group = 0
    mask_array = [0] * total_len
    for i, len_ in enumerate(word_lens):
      mask_array[start] = group + 1
      group += 1
    #     print(i, mask_array)
      if not tokens:
        start += len_
      else:
        start_ = start
        len__ = 0
        while len__ != len_:
          len__ += len(tokens[start_])
          start_ += 1
    #         print(i, start_, len__, len_)
        start = start_
      
    for i in range(total_len):
      if mask_array[i] == 0:
        mask_array[i] = mask_array[i - 1]
  except Exception:
    mask_array = [x + 1 for x in range(total_len)]
  return mask_array

def fix_words_mask(mask_array):
  delta = 0
  for i in range(len(mask_array)):
    if i > 1:
      delta = mask_array[i] - mask_array[i - 1]
      if delta > 1:
        for j in range(i, len(mask_array)):
          mask_array[j] -= (delta - 1)
        break
  return mask_array

try:
  from gensim.models.callbacks import CallbackAny2Vec

  class MonitorCallback(CallbackAny2Vec):
    def __init__(self, name):
      self.name = name
      self.epoch = 1
      self.timer = gezi.Timer()
    
    def on_epoch_end(self, model):
      # TODO 为什么打印train loss一直是0
      print('name:', self.name, 'epoch:', self.epoch, "model loss:", model.get_latest_training_loss(), f'elapsed minutes: {self.timer.elapsed_minutes():.2f}')  # print loss
      self.epoch += 1
except Exception:
  pass
