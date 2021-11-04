#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict, ChainMap
import glob
import sys 
import functools
sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')
import dask.dataframe as dd
from multiprocessing import Pool, Manager, cpu_count
from joblib import Parallel, delayed
from collections import Counter
import pymp
from icecream import ic
import gezi
from gezi import tqdm
tqdm.pandas()


# In[16]:


timer = gezi.Timer('read user_action2.feather', True)
d = pd.read_feather('../input/user_action2.feather')
d = d.sort_values(['date_'], ascending=False)
d = d.astype({'feedid': int, 'userid': int, 'date_': int})
timer.print()


# In[17]:


d.userid.min()


# In[18]:


d.head()


# In[19]:


d[(d.date_ == 14) & (d.actions == 0)].head()


# In[20]:


df = d


# In[21]:


d.userid.describe()


# In[22]:


ACTIONS = [
  'read_comment',
  'comment',
  'like',
  'click_avatar',
  'forward',
  'follow',
  'favorite'
]

ACTIONS2 = ACTIONS + ['finish', 'stay']

HIS_ACTIONS = ACTIONS2 + [
    'pos', 'neg', 'unfinish', 'unstay', \
    #     'unfinish2', 'unstay2', 
#     'show', 
#   'pos_action', 'neg_action', 'show_action'
]


# In[23]:


def get_history(day, key1, key2):
  d = df[df.date_ < day]
  dfs = {}
  t = tqdm(ACTIONS)
  for action in t:
    t.set_postfix({'action': action})
    dfs[action] = d[d[action] == 1].groupby([key1])[key2].progress_apply(list).reset_index(name=action)
  dfs['finish'] = d[d.finish_rate > 0.99].groupby([key1])[key2].progress_apply(list).reset_index(name='finish')
  dfs['unfinish'] = d[d.finish_rate < 0.01].groupby([key1])[key2].progress_apply(list).reset_index(name='unfinish')
  dfs['stay'] = d[d.stay_rate > 1].groupby([key1])[key2].progress_apply(list).reset_index(name='stay')
  dfs['unstay'] = d[d.stay_rate < 0.01].groupby([key1])[key2].progress_apply(list).reset_index(name='unstay')
  dfs['neg'] = d[d.actions == 0].groupby([key1])[key2].progress_apply(list).reset_index(name='neg')
  dfs['pos'] = d[d.actions > 0].groupby([key1])[key2].progress_apply(list).reset_index(name='pos')
  return dfs


# In[24]:


def convert(dfs, key='feedid'):
  history = {}
  if key == 'feedid':
    vocab = gezi.Vocab('../input/doc_vocab.txt')
    vocab2 = gezi.Vocab('../input/user_vocab.txt')
  else:
    vocab = gezi.Vocab('../input/user_vocab.txt')
    vocab2 = gezi.Vocab('../input/doc_vocab.txt')
  for i in range(vocab.size()):
    if i < 2:
      continue
    history[int(vocab.key(i))] = {}
    for action in HIS_ACTIONS:
       history[int(vocab.key(i))][action] = []
  for action in tqdm(HIS_ACTIONS):
#     ic(action)
    d = dfs[action]
    for row in d.itertuples():
      row = row._asdict()
      history[row[key]][action] = [vocab2.id(x) for x in row[action]]
  return history
      


# In[25]:


# 二度关系 通过doc 找 对应的 user 再集合 这些user 最热门的doc
def convert2(dfs, dic, key='feedid'):
  history = {}
  if key == 'feedid':
    vocab = gezi.Vocab('../input/doc_vocab.txt')
    vocab2 = gezi.Vocab('../input/user_vocab.txt')
  else:
    vocab = gezi.Vocab('../input/user_vocab.txt')
    vocab2 = gezi.Vocab('../input/doc_vocab.txt')
#     actions = HIS_ACTIONS
  actions = ['comment', 'follow', 'favorite']
  for i in range(vocab.size()):
    if i < 2:
      continue
    history[int(vocab.key(i))] = {}
    for action in actions:
       history[int(vocab.key(i))][action] = []
  for action in tqdm(actions):
#     ic(action)
    d = dfs[action]
    for row in d.itertuples():
      row = row._asdict()
      userids = row[action]
      counter = Counter()
      for userid in userids:
        for docid in dic[userid][action]:
          counter[docid] += 1
      history[row[key]][action] = [docid for docid, count in counter.most_common(100)]
  return history
      


# In[26]:


def get_feed_history(day):
  return get_history(day, 'feedid', 'userid') 


# In[27]:


def get_user_history(day):
  return get_history(day, 'userid', 'feedid')


# In[29]:


# dfs_feed = get_feed_history(15)


# In[30]:


# dfs_feed['comment']['count'] = dfs_feed['comment'].comment.apply(len)


# In[31]:


# dfs_feed['comment'].describe()


# In[32]:


# dfs_user = get_user_history(14)


# In[33]:


# dfs_user['comment']


# In[34]:


# len(dfs_user['comment'][dfs_user['comment'].userid == 149389].comment.values[0])


# In[35]:


# feed_his = convert(dfs_feed, 'feedid')


# In[36]:


# feed_his[1]


# In[37]:


# dfs_feed['read_comment'].feedid.min()


# In[38]:


DAYS = 15
nw = cpu_count()
nw = min(nw, DAYS)
if 'tione' in os.environ['PATH']:
  for day in tqdm(range(DAYS)):
    day += 1
    dfs_feed = get_feed_history(day)
    # dfs_user = get_user_history(day)
    # user_his = convert(dfs_user, 'userid')
    # feed_his = convert2(dfs_feed, user_his, 'feedid')
    feed_his = convert(dfs_feed, 'feedid')
    gezi.save_pickle(feed_his, f'../input/feed_history_{day}.pkl')
else:
  with pymp.Parallel(nw) as p:
    for day in p.range(DAYS):
      day += 1
      dfs_feed = get_feed_history(day)
      # dfs_user = get_user_history(day)
      # user_his = convert(dfs_user, 'userid')
      # feed_his = convert2(dfs_feed, user_his, 'feedid')
      feed_his = convert(dfs_feed, 'feedid')
      gezi.save_pickle(feed_his, f'../input/feed_history_{day}.pkl')


# In[2]:


his = gezi.read_pickle(f'../input/feed_history_15.pkl')


# In[10]:


his[list(his.keys())[300]]


# In[ ]:




