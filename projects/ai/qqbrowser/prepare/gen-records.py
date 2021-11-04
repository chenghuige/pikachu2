#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-records.py
#        \author   chenghuige  
#          \date   2021-08-26 14:47:28.786761
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')
import os

import random
from multiprocessing import Pool, Manager, cpu_count
import pymp

from absl import app, flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('num_records', 40, '')
flags.DEFINE_integer('buf_size', 1000, '')
flags.DEFINE_integer('index', None, '')
flags.DEFINE_integer('seed_', 1024, '')
flags.DEFINE_bool('force_shuffle', True, '')
flags.DEFINE_integer('ori', 1, '')

import gezi
from gezi import tqdm
import melt as mt

from qqbrowser.dataset import Dataset
from qqbrowser import config
from qqbrowser.config import *  

record_files = None
tag_vocab = None
cat_vocab = None
subcat_vocab = None

def build_features(index):
  num_examples = {
    'train': FLAGS.num_train,
    'valid': FLAGS.num_valid,
    # 'test': FLAGS.num_test,
    # 'test_a': FLAGS.num_test,
    # 'test_b': FLAGS.num_test
  }
  
  # might use valid for train, so shuffle it 
  shuffle = False if 'test' in FLAGS.mode else True
  shuffle_ = shuffle
  # 如果多个文件 不用shard by files 然后 又shuffle 那么会比较慢 如果不是force_shuffle就取消
  # 如果shard by files 需要files 和 进程数对齐 否则也比较慢
  if len(record_files) > 1 and not FLAGS.force_shuffle:
    shuffle_ = False
  if FLAGS.mode in num_examples:
    dataset = Dataset('valid', files=record_files, num_instances=num_examples[FLAGS.mode])
  else:
    dataset = Dataset('valid', files=record_files, recount=True)
  datas = dataset.make_batch(512, world_size=FLAGS.num_records, rank=index, 
                             repeat=False, drop_remainder=False, 
                             shuffle=shuffle_,
                             shard_by_files=False, return_numpy=True)
  num_steps = -(-dataset.num_steps // FLAGS.num_records)
  out_dir = f'../input/{FLAGS.records_name}/{FLAGS.mode}'
  if index == 0:
    gezi.try_mkdir(out_dir)
    # 注意不要用ic 这里多进程不安全
    print('-----------------------------------------------')
    print(record_files, dataset.num_instances, dataset.num_steps, FLAGS.num_records, num_steps, out_dir, shuffle_, shuffle)
    print('-----------------------------------------------')
  ofile = f'{out_dir}/{index}.tfrec'
  keys = []
  
  print('out_dir', out_dir, 'ofile', ofile)
  with mt.tfrecords.Writer(ofile, buffer_size=FLAGS.buf_size, shuffle=shuffle, seed=FLAGS.seed_) as writer:
    for x, _ in tqdm(datas, total=num_steps):
      if not keys:
        keys = list(x.keys())
      count = len(x[keys[0]])
      for i in range(count):
        fe = {}
        for key in keys:
          if key == 'frames':
            fe[key] = list(x[key][i].reshape(-1))
          elif key == 'tag_id':
            tags = [tag_vocab.id(tag, 0) for tag in x[key][i]]
            # tags = list(x[key][i])
            fe[key] = gezi.pad(tags, FLAGS.max_tags)
          elif key == 'cat':
            cat = cat_vocab.id(x[key][i][0])
            fe[key] = [cat]
          elif key == 'subcat':
            subcat = subcat_vocab.id(x[key][i][0])
            fe[key] = [subcat]
          else:
            fe[key] = list(x[key][i])
            
        fe['ori'] = [FLAGS.ori]
        writer.write_feature(fe)

def main(_):
  FLAGS.mode = FLAGS.mode or 'valid'
  FLAGS.seed = FLAGS.seed_
  
  marks = {
    'train': 'pointwise',
    'valid': 'pairwise',
    'test': 'test',
    'test_a': 'test_a',
    'test_b': 'test_b',
  }
  mark = marks[FLAGS.mode]
  
  FLAGS.parse_strategy = 1
  FLAGS.buffer_size = 100
  config.init()
  FLAGS.dump_records = True
  FLAGS.use_title = True
  FLAGS.use_asr = True
  FLAGS.use_frames = True
  FLAGS.dynamic_pad = True
  
  random.seed(FLAGS.seed_)
  
  global record_files, tag_vocab, cat_vocab, subcat_vocab
  record_files = gezi.list_files(f'../input/{mark}/*.tfrecords') 
  ic(record_files)
  random.shuffle(record_files)
  ic(record_files)
  tag_vocab = gezi.Vocab('../input/tag_vocab.txt')
  cat_vocab = gezi.Vocab('../input/cat_vocab.txt')
  subcat_vocab = gezi.Vocab('../input/subcat_vocab.txt')
    
  if FLAGS.num_records == 1 or FLAGS.index is not None:
    build_features(FLAGS.index or 0)
  else:
    # with pymp.Parallel(FLAGS.num_records) as p:
    #   for i in p.range(FLAGS.num_records):
    #     build_features(i)
    nw = min(cpu_count(), FLAGS.num_records)
    with Pool(nw) as p:
      p.map(build_features, range(FLAGS.num_records))

#   # TODO FIXME 查一下哪里写了这个num_records.txt 不正确的数值而且。。 有点诡异
#   ic| util.py:2531 in get_num_records_from_dir()
#     'num_records from ': 'num_records from '
#     num_records_file: '../input/tfrecords3/valid/num_records.txt'
#     num_records: 57249
# ic| gen-records.py:147 in main()
#     out_dir: '../input/tfrecords3/valid'
#     mt.get_num_records_from_dir(out_dir): 57249
# ic| gen-records.py:148 in main()
#     out_dir: '../input/tfrecords3/valid'
#     mt.get_num_records_from_dir(out_dir, recount=True): 63613
#  python gen-records.py --mode=valid --rv=3 --merge_text
#  python gen-records.py --mode=valid --rv= #这个结果就正常 也生成了num_records.txt 但是是63613正确数值
  out_dir = f'../input/{FLAGS.records_name}/{FLAGS.mode}'
  ic(out_dir, mt.get_num_records_from_dir(out_dir))
  ic(out_dir, mt.get_num_records_from_dir(out_dir, recount=True))
  # HACK here
  gezi.try_remove(f'{out_dir}/num_records.txt')


if __name__ == '__main__':
  app.run(main)  
