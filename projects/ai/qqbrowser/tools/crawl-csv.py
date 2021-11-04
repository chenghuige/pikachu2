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

flags.DEFINE_integer('num_ids', 10000, '')

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

def crawl(id):
  url = f'https://kandianshare.html5.qq.com/v3/video/{id}'     
  # ic(url)  
  try:
    res = requests.get(url=url, timeout=2)        
  except Exception:
    # ic(id)
    return {}
  # ic(req.text)          
  html = res.text        
  dom = etree.HTML(html)
  # ic(dom)
  hrefs = dom.xpath(u"//script[@id='config']")
  if len(hrefs) != 1:
    ic(id, len(hrefs))
    ic([(i, x.text) for i, x in enumerate(hrefs)])
    return {}
  text = hrefs[0].text[len('window._configs='):]
  m = json.loads(text)
  # ic(m)
  # ic(list(m.keys()))
  keys = ['vid', 'title', 'tags', 'imageUrl', 'mediaInfo', 'subject', 'type', 'totalTime']

  x = OrderedDict()
  m = m['videoContent']
  for key in keys:
    if key == 'vid':
      x['id'] = m[key]
    elif key == 'mediaInfo':
      x['mediaName'] = m[key]['mediaName']
    elif key == 'tags':
      x['tags'] = ','.join([x['sTagName'] for x in m['tags']])
      x['tag_weights'] = ','.join([str(x['fWeight']) for x in m['tags']])
    else:
      x[key] = m[key]
  return x

def crawls(ids):
  l = []
  t = tqdm(enumerate(ids), total=len(ids))
  for i, id in t:
    x = crawl(id)
    if x:
      l.append(x)

    t.set_postfix({'ok': len(l) / (i + 1)})  
  return l

def deal(ids):
  nw = cpu_count()
  ids_list = np.array_split(ids, nw)
  res = Manager().dict()
  with pymp.Parallel(nw) as p:
    for i in p.range(nw):
      res[i] = crawls(ids_list[i])

  l = []
  for i in res:
    l.extend(res[i])

  return l

def main(_):
  df = pd.read_csv('../input/ids.csv')
  all_ids = df.id.values
  total = len(all_ids)  

  finished_ids = set()
  if os.path.exists('../input/info.csv'):
    df = pd.read_csv('../input/info.csv')
    finished_ids = set(df.id.values)
    all_ids = [x for x in all_ids if x not in finished_ids]
    
  ids = all_ids

  failed_ids = set()

  i = 0
  with tqdm(total=total) as pbar:
    pbar.update(len(finished_ids))
    while True:
      if len(ids) == 0:
        break
      ic(i, len(ids), 1. - len(ids) / total)

      if FLAGS.num_ids:
        ids = ids[:FLAGS.num_ids]

      l = deal(ids)

      ok_ids = set([x['id'] for x in l])
      for id in ids:
        if id not in ok_ids:
          failed_ids.add(id)

      if l:
        pd.DataFrame(l).to_csv('../input/info.csv', index=False, mode='a', header=not os.path.exists('../input/info.csv'))

      ic(i, len(l), len(l)/ len(ids))
      i += 1
      pbar.update(len(l))
      time.sleep(1)

      ids = all_ids
      if os.path.exists('../input/info.csv'):
        df = pd.read_csv('../input/info.csv')
        finished_ids = set(df.id.values)
        ids1 = [x for x in ids if (x not in finished_ids) and (x not in failed_ids)]
        np.random.shuffle(ids1)
        ids2 = [x for x in ids if (x not in finished_ids) and (x in failed_ids)]
        np.random.shuffle(ids2)
        ic(len(ids1), len(ids2), len(ids1) + len(ids2), len(ids2) / (len(ids1) + len(ids2)))
      ids = ids1 + ids2

if __name__ == '__main__':
  app.run(main)  
  