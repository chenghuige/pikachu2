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
import pymp
import gezi
# from gezi import tqdm
from tqdm import tqdm
tqdm.pandas()


# In[2]:


timer = gezi.Timer('read user_action2.feather', True)
d = pd.read_feather('../input/user_action2.feather')
d.feedid = d.feedid.astype(int)
d.userid = d.userid.astype(int)
d.date_ = d.date_.astype(int)
timer.print()


# In[3]:


# d.head()


# In[8]:


# d[(d.date_ == 14) & (d.actions == 0)]


# In[9]:


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


# In[10]:


def is_neg(row):
  return row['actions'] == 0

d = d.sort_values(['date_'], ascending=False)
# TODO 如果能看到test集合 那么需要先把test集合的shows加上

userids = [int(x.strip().split()[0]) for x in open('../input/user_vocab.txt').readlines()]
print('num_userids', len(userids))
history = {}
for userid in userids:
  history[userid] = {}
  for action in HIS_ACTIONS:
    history[userid][action] = []

def gen_history_row(row, history):
  feedid = int(row['feedid'])
  userid = int(row['userid'])
  day = int(row['date_'])
  feedid = (feedid, day)
  his = history[userid]
#   his['show'].append(feedid)
  if row['finish_rate'] > 0.99:
    his['finish'].append(feedid)
  if row['finish_rate'] < 0.01:
    his['unfinish'].append(feedid)
#   if row['finish_rate'] < 0.1:
#     his['unfinish2'].append(feedid)
  if row['stay_rate'] > 1:
    his['stay'].append(feedid)
  if row['stay_rate'] < 0.01:
    his['unstay'].append(feedid)
#   if row['stay_rate'] < 0.1:
#     his['unstay2'].append(feedid)
  actions = [0] * len(ACTIONS2)
  for i, action in enumerate(ACTIONS):
    if row[action] > 0:
      his[action].append(feedid)
      actions[i] = 1
  actions[-2] = row['finish_rate']
  actions[-1] = row['stay_rate']
  is_neg_row = is_neg(row)
  if not is_neg_row:
    his['pos'].append(feedid)
#     his['pos_action'].append(actions)
  else:
    his['neg'].append(feedid)

#   his['show_action'].append(actions)

def gen_history(d, userids=None):
  history = {}
  if userids is not None:
    d = d[d.userid.isin(set(userids))]
  else:
    userids = set(d.userid)
  for userid in userids:
    history[userid] = {}
    for action in HIS_ACTIONS:
      history[userid][action] = []

  for row in tqdm(d.itertuples(), total=len(d), desc=f'user_action users:{len(userids)}'):
    row = row._asdict()
    gen_history_row(row, history)
  return history  
    
# nw = 12
# res = Parallel(n_jobs=nw)(delayed(lambda x: gen_history(d, x))(uids) for uids in np.array_split(userids, nw))
# print(res)
# d.progress_apply(gen_history_row, axis=1)
# d.parallel_apply(gen_history_row, axis=1)

## 可能需要meta 较为麻烦
# d2 = dd.from_pandas(d,npartitions=40)

nw = cpu_count()
userids_list = np.array_split(userids, nw)
res = Manager().dict()
with pymp.Parallel(nw) as p:
  for i in p.range(nw):
    res[i] = gen_history(d, userids_list[i])
with gezi.Timer('ChainMap merge res', True):
  history = dict(ChainMap(*res.values()))

# nw = cpu_count()
# userids_list = np.array_split(userids, nw)
# pfunc = functools.partial(gen_history, d=d)
# with Pool(nw) as p:
#   res = p.map(pfunc, userids_list)
# with timer = gezi.Timer('ChainMap merge res', True):
#   history = dict(ChainMap(*res))


# In[11]:


len(history)


# In[12]:


# history[userids[0]]


# In[13]:


with gezi.Timer('save history.pkl', True):
  gezi.save_pickle(history, '../input/history.pkl')


# In[ ]:





# In[14]:


m = {
  'userid': [],
  'pos': [],
  'pos_len': [],
  'finish': [],
  'finish_len': [],
  'stay': [],
  'stay_len': [],
  'neg': [],
  'neg_len': [],
  'unfinish': [],
  'unfinish_len': [],  
  'unstay': [],
  'unstay_len': [], 
#   'unfinish2': [],
#   'unfinish2_len': [],  
#   'unstay2': [],
#   'unstay2_len': [], 
#   'latest': [],
#   'latest_len': [],
#   'show': [],
#   'show_len': [],
}
for action in ACTIONS:
  m[action] = []
  m[f'{action}_len'] = []

for userid in tqdm(userids):
  m['userid'].append(userid)
  his = history[userid]
  for action in HIS_ACTIONS:
    if action in m:
      m[action].append(' '.join(map(str, his[action])))
      m[f'{action}_len'].append(len(his[action]))


# In[15]:


his_df = pd.DataFrame(m)


# In[ ]:


# his_df.head()


# In[ ]:


# his_df[his_df.userid == 99097].unfinish_len


# In[16]:


his_df.describe(percentiles=[.25,.5,.75,.9,.95,.99, .999])


# In[17]:


his_df.reset_index().to_feather('../input/history.feather')


# In[ ]:




