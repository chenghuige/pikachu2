#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   visualize.py
#        \author   chenghuige  
#          \date   2021-08-31 11:56:15.721949
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')


from absl import app, flags
FLAGS = flags.FLAGS

import random
from itertools import count
import heapq as hq
import multiprocessing

import glob
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
import gezi
from gezi import tqdm

from qqbrowser.config import *

MAX_SHOW_LEN = 15
def wrap_str(title):
  import textwrap
  return '\n'.join(textwrap.wrap(title, MAX_SHOW_LEN))

def limit_str(text, max_len, last_tokens, sep='|'):
  if len(text) <= max_len:
    return text
  first_tokens = max(max_len - last_tokens - len(sep), 1)
  return text[:first_tokens] + sep + text[-last_tokens:]

class ImageLogger():
  def __init__(self, step=0, num_random=10, num_worst=10, num_best=5, load_tags=False, seed=None, path=None):
    self.best_imgs = []
    self.worst_imgs = []
    self.random_imgs = []
    
    self.num_random = num_random
    self.num_worst = num_worst
    self.num_best = num_best

    np.random.seed(seed)

    self.step = step
    self.tiebreaker = count()
    self.logger = gezi.get_summary_writer(path)
    
    # 抓取的信息 包括文本tag表示
    self.info = pd.read_csv('../input/info/info.csv')
    # 原始信息
    self.infos = pd.read_csv('../input/info/infos.csv')
    # ic(len(self.info), len(self.infos))
    self.info = self.info.set_index('id')
    self.infos = self.infos.set_index('id')
    
    self.tag_vocab = gezi.Vocab('../input/tag_vocab.txt')

    # with gezi.Timer('read tag names'):
    tag_names = gezi.read_pickle('../input/tag_names.pkl')
    self.tag_names = {}
    for tag in tag_names:
      self.tag_names[str(tag)] = tag_names[tag].most_common(2)
    
    if load_tags:
      try:
        self.tags = gezi.read_pickle(f'{path}/top_tags.pkl')
        self.tag_weights = gezi.read_pickle(f'{path}/top_weights.pkl')
      except Exception as e:
        ic(e)
    
  def get_text(self, vid, relevance=-1, similarity=-1, top_tag=None, top_weight=None):
    info, infos = self.info, self.infos
    tags = wrap_str(str(info.loc[vid].tags))
    title = wrap_str(str(infos.loc[vid].title))
    catid = infos.loc[vid].category_id
    tagid = wrap_str(str(infos.loc[vid].tag_id))
    asr = wrap_str(limit_str(str(infos.loc[vid].asr_text), 128, 10))
    if top_tag is None:
      # text = f'id:{vid}\ncat:{catid} label:[{relevance:.2f}] pred:[{similarity:.2f}]\n-------------------------\n{tags}\n{tagid}\n-------------------------\n{asr}\n-------------------------\n{title}'
      text = f'id:{vid}\ncat:{catid} label:[{relevance:.2f}] pred:[{similarity:.2f}]\n-------------------------\n{tags}\n-------------------------\n{asr}\n-------------------------\n{title}'
    else:
      top_tag = [self.tag_vocab.key(x) for x in top_tag]
      if top_weight is None:
        # top_tag = wrap_str(','.join([self.tag_vocab.key(x) for x in top_tag]))
        top_tag = [f'{self.tag_names.get(tag, "")}' for tag in top_tag]
        top_tag = '\n'.join()
      else:
        # top_tag = [f'{tag}:{self.tag_names.get(tag, "")}:{weight:.2f}' for tag, weight in zip(top_tag, top_weight)]
        # top_tag = wrap_str(','.join(top_tag))
        top_tag = [f'{self.tag_names.get(tag, "")}:{weight:.2f}' for tag, weight in zip(top_tag, top_weight)]
        top_tag = '\n'.join(top_tag)
      # text = f'id:{vid}\ncat:{catid} label:[{relevance:.2f}] pred:[{similarity:.3f}]\n-------------------------\n{tags}\n{tagid}\n-------------------------\n{top_tag}\n-------------------------\n{asr}\n-------------------------\n{title}'
      text = f'id:{vid}\ncat:{catid} label:[{relevance:.2f}] pred:[{similarity:.3f}] \n-------------------------\n{tags}\n-------------------------\n{top_tag}\n-------------------------\n{asr}\n-------------------------\n{title}'
    return text
  
  def has_show(self, left, right):
    info, infos = self.info, self.infos
    if not (left in info.index and right in info.index):
      return False
    img1 = f'../input/imgs/{left}.jpg'
    img2 = f'../input/imgs/{right}.jpg'
    if not (os.path.exists(img1) and os.path.exists(img2)):
      return False
    return True

  def display_pair(self, left, right, relevance=0, similarity=0, top_tag=None, top_weight=None):
    img1 = f'../input/imgs/{left}.jpg'
    img2 = f'../input/imgs/{right}.jpg'
    if top_tag is None:
      try:
        top_tag1, top_tag2 = self.tags[str(left)], self.tags[str(right)]
      except Exception:
        top_tag1, top_tag2 = None, None
    else:
      top_tag1, top_tag2 = top_tag 
    if top_weight is None:
      try:
        top_weight1, top_weight2 = self.tag_weights[str(left)], self.tag_weights[str(right)]
      except Exception:
        top_weight1, top_weight2 = None, None
    else:
      top_weight1, top_weight2 = top_weight
    text1 = self.get_text(left, relevance, similarity, top_tag1, top_weight1)
    text2 = self.get_text(right, relevance, similarity, top_tag2, top_weight2)
    return gezi.plot.display_images([img1, img2], titles=[text1, text2], spacing=0.05, img2bytes=False)
    # return gezi.plot.display_images([img1, img2], titles=[text1, text2], spacing=0.05, img2bytes=True)

  def log(self, queries, candidates, similarities, relevances=None, top_tags=None, top_weights=None):
    self._log(queries, candidates, similarities, relevances, top_tags, top_weights)
    ## will hang.. TODO
    # p = multiprocessing.Process(target=self._log, args=(queries, candidates, similarities, relevances))
    # p.start()
    
  def _log(self, queries, candidates, relevances, similarities, top_tags=None, top_weights=None):
    ic(relevances, similarities)
    assert relevances is not None or similarities is not None
    if relevances is None:
      relevances = np.ones_like(similarities) * -1.
    if similarities is None:
      similarities = np.ones_like(relevances) * -1.

    if top_tags is None:
      top_tags = [None] * len(relevances)
    if top_weights is None:
      top_weights = [None] * len(relevances)
      
    queries = np.asarray(queries).astype(np.int64)
    candidates = np.asarray(candidates).astype(np.int64)

    # random_img_indexes = np.asarray(range(len(queries)))
    # np.random.shuffle(random_img_indexes)
    # random_img_indexes = set(random_img_indexes[:self.num_random])
      
    for i, (query, candidate, similarity, relevance, top_tag, top_weight) in tqdm(enumerate(zip(queries, candidates, similarities, relevances, top_tags, top_weights)), total=len(queries), desc='prepare log images', leave=False):
      if not self.has_show(query, candidate):
        continue
      metric = (similarity - relevance) ** 2
      item = [-metric, next(self.tiebreaker), (query, candidate, relevance, similarity, top_tag, top_weight)]
      item2 = [metric, next(self.tiebreaker), *item[2:]]
      hq.heappush(self.best_imgs, item)
      hq.heappush(self.worst_imgs, item2)
      if len(self.best_imgs) > self.num_best:
        hq.heappop(self.best_imgs)
      if len(self.worst_imgs) > self.num_worst:
        hq.heappop(self.worst_imgs)
      
      # if i in random_img_indexes:
      if len(self.random_imgs) < self.num_random:
        self.random_imgs.append(item)
    
    # ic(len(self.random_imgs), len(self.worst_imgs), len(self.best_imgs))
    self.tb_images(self.random_imgs, self.worst_imgs, self.best_imgs)
    
  def _tb(self, item, tag, index):
    query, candidate, relevance, similarity, top_tag, top_weight = item[-1]
    image = self.display_pair(query, candidate, relevance, similarity, top_tag, top_weight)
    title = ':'.join(map(str, [relevance, similarity]))
    # self.logger.image(f'{tag}/{index}', image, self.step, title=title)
    self.logger.log_image(f'{tag}/{index}', image, self.step, title=title)

  def tb_images(self, random_imgs, worst_imgs, best_imgs):  
    for i, img in tqdm(enumerate(random_imgs), desc='write_random_imgs', total=len(random_imgs), leave=False):
      self._tb(img, 'random', i)
    
    for i, img in tqdm(enumerate(worst_imgs), desc='write_worst_imgs', total=len(worst_imgs), leave=False):
      self._tb(img, 'worst', i)
    
    for i, img in tqdm(enumerate(best_imgs), desc='write_best_imgs', total=len(best_imgs), leave=False):
      self._tb(img, 'best', i)

def main(_):
  df = pd.read_csv('../working/11/now/valid.csv')
  ic(len(df))
  queries, candidates, relevances_, similarities_ = df['query'].values, df.candidate.values, df.relevance.values, df.similarity.values
  ImageLogger(0, FLAGS.num_random, FLAGS.num_worst, 
              FLAGS.num_best, seed=FLAGS.vis_seed, path='../working/tmp') \
        .log(queries, candidates, relevances_, similarities_)
  print('done')

if __name__ == '__main__':
  app.run(main)  