#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-vocabs.py
#        \author   chenghuige  
#          \date   2020-06-12 15:28:22.032538
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os


# coding: utf-8

# In[2]:


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

os.system('mkdir -p ../input/all')

# # vocab of watch vids

# In[ ]:


counter_train = gezi.WordCounter()
counter_eval = gezi.WordCounter()
counter = gezi.WordCounter()


# In[ ]:


for k in tqdm(range(30)):
  d = gezi.read_parquet(f'../input/train/part_{k + 1}/user.parquet')
  d.watch = d.watch.apply(json.loads)
  watches = d.watch.values
  for i in range(len(watches)):
    for j in range(len(watches[i])):
      counter.add(watches[i][j][1])
      counter_train.add(watches[i][j][1])


# In[ ]:


d = gezi.read_parquet('../input/eval/user.parquet')
d.watch = d.watch.apply(json.loads)
watches = d.watch.values
for i in tqdm(range(len(watches))):
  for j in range(len(watches[i])):
    counter.add(watches[i][j][1])
    counter_eval.add(watches[i][j][1])


counter.save('../input/all/watch_vids.txt')


# In[ ]:


counter_train.save('../input/train/watch_vids.txt')


# In[ ]:


counter_eval.save('../input/eval/watch_vids.txt')


# # vocab of vids

# In[ ]:


counter2_train = gezi.WordCounter()
counter2_eval = gezi.WordCounter()
counter2 = gezi.WordCounter()
counter_prev = gezi.WordCounter()


# In[ ]:


for i in tqdm(range(30)):
  d = gezi.read_parquet(f'../input/train/part_{i+1}/context.parquet')
  for vid in d.vid.values:
    counter2.add(vid)
    counter2_train.add(vid)
  d = d[d.prev!=0]
  counter_prev.adds(d.prev)

# In[ ]:


d = gezi.read_parquet(f'../input/eval/context.parquet')
for vid in tqdm(d.vid.values):
  counter2.add(vid)
  counter2_eval.add(vid)
d = d[d.prev!=0]
counter_prev.adds(d.prev)

counter2.save('../input/all/context_vids.txt')

# In[ ]:


counter2_train.save('../input/train/context_vids.txt')


# In[ ]:


counter2_eval.save('../input/eval/context_vids.txt')

couter_prev.save('../input/all/prev_vids.txt')


# In[ ]:


for vid, count in counter2.counter.items():
  counter.add(vid, count)
for vid, count in counter_prev.counter.items():
  counter.add(vid, count)


# In[ ]:


counter.save('../input/all/vids.txt')


# In[ ]:


for vid, count in counter2_train.counter.items():
  counter_train.add(vid, count)


# In[ ]:


counter_train.save('../input/train/vids.txt')


# # vocab of words

# In[ ]:


counter_words = gezi.WordCounter()


# In[ ]:


vinfo = pq.read_table('../input/train/raw.parquet').to_pandas()


# In[ ]:


for i in range(len(vinfo)):
  for word in vinfo.title[i].split(','):
    if word:
      counter_words.add(word)


counter_words.save('../input/all/words.txt')


# # vocab of stars

# In[ ]:


counter = gezi.WordCounter()
counter_train = gezi.WordCounter()
counter_eval = gezi.WordCounter()


# In[ ]:


for k in tqdm(range(30)):
  d = gezi.read_parquet(f'../input/train/part_{k + 1}/item.parquet')
  stars = d.stars.values
  for i in range(len(stars)):
    for star in stars[i]:
      counter.add(star)
      counter_train.add(star)


# In[ ]:


d = gezi.read_parquet('../input/eval/item.parquet')
stars = d.stars.values
for i in range(len(stars)):
  for star in stars[i]:
    counter.add(star)
    counter_eval.add(star)


counter.save('../input/all/stars.txt')


# # vocab fo video classes

# In[3]:


dis = []
for k in tqdm(range(30)):
  d = gezi.read_parquet(f'../input/train/part_{k + 1}/item.parquet')
  dis += [d]
dis += [gezi.read_parquet('../input/eval/item.parquet')]
dis = pd.concat(dis)


# In[ ]:


counter = gezi.WordCounter()
counter.adds(dis.cid)
counter.save('../input/all/cid.txt')


# In[ ]:


counter = gezi.WordCounter()
counter.adds(dis.class_id)
counter.save('../input/all/class_id.txt')


# In[ ]:


counter = gezi.WordCounter()
counter.adds(dis.second_class)
counter.save('../input/all/second_class.txt')


# In[5]:


counter = gezi.WordCounter()
counter.adds(dis.is_intact)
counter.save('../input/all/is_intact.txt')


# # vocab of class infos

# In[ ]:


dcs = []
for k in tqdm(range(30)):
  d = gezi.read_parquet(f'../input/train/part_{k + 1}/context.parquet')
  dcs += [d]
dcs += [gezi.read_parquet('../input/eval/context.parquet')]
dcs = pd.concat(dcs)


counter = gezi.WordCounter()
counter.adds(dis.vid)
counter.save('../input/all/vid.txt')


# In[ ]:


counter = gezi.WordCounter()
counter.adds(dcs.did)
counter.save('../input/all/did.txt')


# In[ ]:


counter = gezi.WordCounter()
counter.adds(dcs.aver)
counter.save('../input/all/aver.txt')


# In[ ]:


counter = gezi.WordCounter()
counter.adds(dcs.mf)
counter.save('../input/all/mf.txt')


# In[ ]:


counter = gezi.WordCounter()
counter.adds(dcs['mod'])
counter.save('../input/all/mod.txt')


# In[ ]:


counter = gezi.WordCounter()
counter.adds(dcs.region)
counter.save('../input/all/region.txt')


# In[ ]:


counter = gezi.WordCounter()
counter.adds(dcs.sver)
counter.save('../input/all/sver.txt')
  
