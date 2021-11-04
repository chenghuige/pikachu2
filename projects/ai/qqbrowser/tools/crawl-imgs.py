#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   head-tfrecord.py
#        \author   chenghuige  
#          \date   2019-09-11 11:00:01.818073
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
from absl import app, flags
from gezi.util import index
FLAGS = flags.FLAGS

flags.DEFINE_integer('num_ids', 1000, '')

import sys 
import os

import time
import requests               
import json
from collections import OrderedDict
from lxml import etree
import numpy as np
import pandas as pd
from multiprocessing import Pool, Manager, cpu_count
import pymp 

import melt as mt
import gezi
from gezi import tqdm

# TODO 可能会hang..

def crawl(id, img_url):
  url = img_url 
  # ic(url)  
  try:
    res = requests.get(url=url, timeout=2)        
  except Exception:
    return {}

  try:
    with open(f'../input/imgs/{id}.jpg','wb') as f:
      # ic(id, img_url, res.content)
      f.write(res.content)
  except Exception as e:
    ic(e)
    return {}

  x = {'id': id}
  return x

def crawls(ids, img_urls):
  l = []
  t = tqdm(enumerate(zip(ids, img_urls)), total=len(ids))
  for i, (id, img_url) in t:
    x = crawl(id, img_url)
    if x:
      l.append(x)

    t.set_postfix({'ok': len(l) / (i + 1)})  
  return l

@gezi.set_timeout(600, lambda: [])
def deal(ids, img_urls):
  nw = cpu_count()
  ids_list = np.array_split(ids, nw)
  url_list = np.array_split(img_urls, nw)
  res = Manager().dict()
  with pymp.Parallel(nw) as p:
    for i in p.range(nw):
      res[i] = crawls(ids_list[i], url_list[i])

  l = []
  for i in res:
    l.extend(res[i])

  return l

def main(_):
  gezi.try_mkdir('../input/imgs')
  df = pd.read_csv('../input/info.csv')
  all_ids = df.id.values
  total = len(all_ids)  

  all_imgs = df.imageUrl.values
  m = dict(zip(all_ids, all_imgs))
 
  finished_ids = set()
  if os.path.exists('../input/imgs.csv'):
    df = pd.read_csv('../input/imgs.csv')
    finished_ids = set(df.id.values)
    all_ids = [x for x in all_ids if x not in finished_ids]
    
  ids = all_ids

  i = 0
  with tqdm(total=total) as pbar:
    pbar.update(len(finished_ids))
    while True:
      if len(ids) == 0:
        break
      ic(i, len(ids), 1. - len(ids) / total)

      if FLAGS.num_ids:
        ids = ids[:FLAGS.num_ids]

      imgs = [m[id] for id in ids]
      l = deal(ids, imgs)
      if l:
        pd.DataFrame(l).to_csv('../input/imgs.csv', index=False, mode='a', header=not gezi.non_empty('../input/imgs.csv'))

      ic(i, len(l), len(l)/ len(ids))
      i += 1
      pbar.update(len(l))
      time.sleep(1)

      ids = all_ids
      if os.path.exists('../input/imgs.csv'):
        df = pd.read_csv('../input/imgs.csv')
        finished_ids = set(df.id.values)
        ids = [x for x in ids if x not in finished_ids]
      np.random.shuffle(ids)

if __name__ == '__main__':
  app.run(main)  
  
