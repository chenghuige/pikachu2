#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   calc.py
#        \author   chenghuige  
#          \date   2020-03-25 21:49:21.445073
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import glob 
import pickle
import pandas as pd
from tqdm import tqdm

import gezi

root = '/search/odin/publicData/CloudS/rank/infos/tuwen/common/'

dirs = glob.glob(f'{root}/*')

pre_dir = None
for dir in dirs:
  if not os.path.exists(f'{dir}/infos.csv'):
    continue
  if not os.path.exists(f'{dir}/doc_hot.pkl'):
    break
  pre_dir = dir
  
docs = {}
if pre_dir:
  with gezi.Timer('load doc pkl', print_fn=print):
    with open(f'{pre_dir}/doc_hot.pkl', 'rb') as f:
      docs = pickle.load(f)

print('docs', len(docs))

users = {}
if pre_dir:
  with gezi.Timer('load user pkl', print_fn=print):
    with open(f'{pre_dir}/user_hot.pkl', 'rb') as f:
      users = pickle.load(f)

print('users', len(users))

with gezi.Timer('read infos.csv', print_fn=print):
  print(f'{dir}/infos.csv')
  d = pd.read_csv(f'{dir}/infos.csv', low_memory=False)

mids = d.mid.values
docids = d.docid.values
durs = d.duration.values

rates = [0.999, 0.99, 0.9]

NUM = 15  

default_dfeats = [0.] * NUM
default_ufeats = [0.] * NUM

DUR, CTR, CTR0, CTR1, CTR2, DUR_COUNTS, DUR_TOTAL, CLICK, CLICK0, CLICK1, CLICK2, SHOW, SHOW0, SHOW1, SHOW2 = range(NUM)

if docs:
  for did in docs:
    dinfo = docs[did]
    assert len(dinfo) == NUM, len(dinfo)
    dinfo[CLICK0] *= rates[0]
    dinfo[CLICK1] *= rates[1]
    dinfo[CLICK2] *= rates[2] 
    dinfo[SHOW0] *= rates[0]
    dinfo[SHOW1] *= rates[1]
    dinfo[SHOW2] *= rates[2] 

if users:
  for uid in users:
    uinfo = users[uid]
    assert len(uinfo) == NUM, len(dinfo)
    uinfo[CLICK0] *= rates[0]
    uinfo[CLICK1] *= rates[1]
    uinfo[CLICK2] *= rates[2] 
    uinfo[SHOW0] *= rates[0]
    uinfo[SHOW1] *= rates[1]
    uinfo[SHOW2] *= rates[2] 
    
uids = set()
dids = set()
for uid, did, dur in tqdm(zip(mids, docids, durs), total=len(mids), ascii=True):
  uids.add(uid)
  dids.add(did)
  click = dur != 0
  dinfo = docs.get(did, default_ufeats.copy())
  uinfo = users.get(uid, default_ufeats.copy())

  if dur > 0:
    dinfo[DUR_COUNTS] += 1
    dinfo[DUR_TOTAL] += dur
    uinfo[DUR_COUNTS] += 1
    uinfo[DUR_TOTAL] += dur

  if click:
    dinfo[CLICK] += 1
    dinfo[CLICK0] += 1
    dinfo[CLICK1] += 1
    dinfo[CLICK2] += 1

    uinfo[CLICK] += 1
    uinfo[CLICK0] += 1
    uinfo[CLICK1] += 1
    uinfo[CLICK2] += 1
    
  dinfo[SHOW] += 1
  dinfo[SHOW0] += 1
  dinfo[SHOW1] += 1
  dinfo[SHOW2] += 1

  uinfo[SHOW] += 1
  uinfo[SHOW0] += 1
  uinfo[SHOW1] += 1
  uinfo[SHOW2] += 1
  
  if did not in docs:
    docs[did] = dinfo
  if uid not in users:
    users[uid] = uinfo

dids_delete = set()
uids_delete = set()

for did in dids:
  dinfo = docs[did]
  dinfo[DUR] = dinfo[DUR_TOTAL] / dinfo[DUR_COUNTS] if dinfo[DUR_COUNTS] else 0.
  dinfo[CTR] = dinfo[CLICK] / dinfo[SHOW]
  dinfo[CTR0] = dinfo[CLICK0] / dinfo[SHOW0]
  dinfo[CTR1] = dinfo[CLICK1] / dinfo[SHOW1]
  dinfo[CTR2] = dinfo[CLICK2] / dinfo[SHOW2]
  # if dinfo[SHOW2]

for uid in uids:
  uinfo = users[uid]
  uinfo[DUR] = uinfo[DUR_TOTAL] / uinfo[DUR_COUNTS] if uinfo[DUR_COUNTS] else 0.
  uinfo[CTR] = uinfo[CLICK] / uinfo[SHOW]
  uinfo[CTR0] = uinfo[CLICK0] / uinfo[SHOW0]
  uinfo[CTR1] = uinfo[CLICK1] / uinfo[SHOW1]
  uinfo[CTR2] = uinfo[CLICK2] / uinfo[SHOW2]



with gezi.Timer('save doc pkl', print_fn=print):
  with open(f'{dir}/doc_hot.pkl', 'wb') as f:
    pickle.dump(docs, f)

with gezi.Timer('save user pkl', print_fn=print):
  with open(f'{dir}/user_hot.pkl', 'wb') as f:
    pickle.dump(users, f)
