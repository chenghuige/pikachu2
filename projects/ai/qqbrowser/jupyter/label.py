#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import sys, os 
sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')
import glob
import numpy as np
import pandas as pd
import gezi
from gezi import tqdm


# %%


label = pd.read_csv('../input/pairwise/label.tsv', names=['vid1', 'vid2', 'relevance'], sep='\t')


# %%


label


# %%


len(set(label.vid1))


# %%


len(set(label.vid2))


# %%


len(set(label.vid1) | set(label.vid2))


# %%


#get_ipython().system('wc -l ../input/pairwise/ids.csv')


# %%


len(set(label.vid1) & set(label.vid2))


# %%


list(set(label.vid1) & set(label.vid2))[:10]


# %%


label[label.vid1 == 1282350513519752622]


# %%


label[label.vid2 == 1282350513519752622]


# %%


FOLDS = 5


# %%


SEED = 521


# %%


intersect_vids = set(label.vid1) & set(label.vid2)


# %%


np.random.seed(SEED)


# %%


vids = list(set(label.vid1))
np.random.shuffle(vids)


# %%


vids = [x for x in vids if x not in intersect_vids]


# %%


vids_list = np.array_split(vids, FOLDS)


# %%


[len(x) for x in vids_list]


# %%


# vids_list[-1] = np.asarray(list(vids_list[-1]) + list(intersect_vids))


# %%


[len(x) for x in vids_list]


# %%


vids_list


# %%


label2 = label[~label.vid1.isin(intersect_vids) & ~label.vid2.isin(intersect_vids)]


# %%


label2


# %%


label3 = label[label.vid1.isin(intersect_vids) | label.vid2.isin(intersect_vids)]


# %%


label3


# %%


# label2 = label


# %%


for i in range(FOLDS):
  d = label2[label2.vid1.isin(set(vids_list[i]))]
  print(len(d))
  d.to_csv(f'../input/pairwise/label_{i}.csv', index=False)


# %%


label3.to_csv(f'../input/pairwise/label_{FOLDS}.csv', index=False)


# %%


len(label)


# %%


len(label2)


# %%


len(vids_list[0])


# %%


label2[label2.vid1.isin(vids_list[0])]


# %%


len(set(label2.vid1))


# %%


vids_list[4]


# %%


ds = {}
for i in range(FOLDS):
  ds[i] = pd.read_csv(f'../input/pairwise/label_{i}.csv')


# %%


def intersect(x, y):
  return len(set(list(x.vid1.values) + list(x.vid2.values)) & set(list(y.vid1.values) + list(y.vid2.values)))


# %%


for i in range(FOLDS):
  for j in range(FOLDS):
      if j > i:
        print(i, j, intersect(ds[i], ds[j]))


# %%


intersect(pd.concat([ds[1],ds[2],ds[3],ds[4]]), ds[0])


# %%


dt = pd.concat([ds[1],ds[2],ds[3],ds[4]])


# %%


len(set(dt.vid1 | dt.vid2))


# %%


len(ds[4])


# %%


ds[0][ds[0].vid1.isin(set(dt.vid1))]


# %%


ds[0][ds[0].vid2.isin(set(dt.vid2))]


# %%


dt


# %%


ds[4]


# %%




