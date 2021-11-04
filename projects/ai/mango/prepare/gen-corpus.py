#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-corpus.py
#        \author   chenghuige  
#          \date   2020-06-12 21:02:41.067102
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import os, sys
import glob
import time
from datetime import timedelta, datetime
import json
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from gezi import tqdm
import gezi

# corpus of vid
with open(f'../input/all/corpus.txt', 'w') as out_all:
  with open(f'../input/all/corpus_novalid.txt', 'w') as out_novalid:
    with open(f'../input/all/corpus_train.txt', 'w') as out_train:
      for k in tqdm(range(30)):
        d = gezi.read_parquet(f'../input/train/part_{k + 1}/user.parquet')
        d.watch = d.watch.apply(json.loads)
        watches = d.watch.values
        with open(f'../input/train/part_{k + 1}/corpus.txt', 'w') as out:
          for i in range(len(watches)):
            l = []
            for j in range(len(watches[i])):
              l.append(str(watches[i][j][1]))
            if l:
              print(' '.join(l), file=out)
              print(' '.join(l), file=out_all)
              if i != 30:
                print(' '.join(l), file=out_novalid)
                print(' '.join(l), file=out_train)
      d = gezi.read_parquet('../input/eval/user.parquet')
      d.watch = d.watch.apply(json.loads)
      watches = d.watch.values
      with open(f'../input/eval/corpus.txt', 'w') as out:
        for i in tqdm(range(len(watches))):
          l = []
          for j in range(len(watches[i])):
            l.append(str(watches[i][j][1]))
          if l:
            print(' '.join(l), file=out)
            print(' '.join(l), file=out_novalid)
            print(' '.join(l), file=out_all)

# corpus of vid without click data
clicks = pd.read_csv('../input/all/clicks.csv')
clicks = set(zip(clicks.did.values, clicks.vid.values))
with open(f'../input/all/corpus_filter.txt', 'w') as out:
  for k in tqdm(range(30)):
    d = gezi.read_parquet(f'../input/train/part_{k + 1}/user.parquet')
    dids = d.did.values
    d.watch = d.watch.apply(json.loads)
    watches = d.watch.values
    for i in range(len(watches)):
      did = dids[i]
      l = []
      for j in range(len(watches[i])):
        vid = watches[i][j][1]
        if not (did, vid) in clicks:
          l.append(str(vid))
      if l:
        print(' '.join(l), file=out)
  d = gezi.read_parquet('../input/eval/user.parquet')
  d.watch = d.watch.apply(json.loads)
  watches = d.watch.values
  for i in tqdm(range(len(watches))):
    l = []
    for j in range(len(watches[i])):
      vid = watches[i][j][1]
      if not (did, vid) in clicks:
        l.append(str(vid))
    if l:
      print(' '.join(l), file=out)

# title
vinfo = pq.read_table('../input/train/raw.parquet').to_pandas()
with open(f'../input/all/corpus_words.txt', 'w') as out:
  titles = vinfo.title.values
  stories = vinfo.story.values
  for i in tqdm(range(len(vinfo))):
    if ','  in titles[i]:
      print(titles[i].replace(',', ' '), file=out)
    if ','  in stories[i]:
      print(stories[i].replace(',', ' '), file=out)

# corpus of stars
with open('../input/all/corpus_stars.txt', 'w') as out:
  for i in tqdm(range(30)):
    d = gezi.read_parquet(f'../input/train/part_{i + 1}/item.parquet')
    for stars in d.stars:
      content = ' '.join(map(str, stars))
      if len(content.strip().split()) > 1:
        print(content, file=out)
  d = gezi.read_parquet(f'../input/eval/item.parquet')
  for stars in d.stars:
    content = ' '.join(map(str, stars))
    if len(content.strip().split()) > 1:
      print(content, file=out)
