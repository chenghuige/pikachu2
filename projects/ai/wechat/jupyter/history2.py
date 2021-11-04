#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict
import glob
import sys 
sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')
import gezi
from gezi import tqdm


# In[2]:


d = pd.read_csv('../input/user_action2.csv')
d = d[d.is_first == 1]
d.feedid = d.feedid.astype(int)
d.userid = d.userid.astype(int)
d.date_ = d.date_.astype(int)


# In[3]:


d.head()


# In[4]:


# get history
# TODO add dislike ? no action and finish rate < 0.1 and stay i
# play time
# staty time as history, play time >  60s play time < 5s like this
# TODO make scripts and parallel base on user
# 历史收益比较大 尝试更多的可能历史

# 引入用户最近观看历史 50 个 ？    每个配合 各种action emb
# latest history  综合表示用户近期历史 ？ TODO

# finish rate 正负 参数如何最好
# stay rate  
# 另外全局角度 actition rate, finish rate, stay rate

ACTIONS = [
  'read_comment',
  'comment',
  'like',
  'click_avatar',
  'forward',
  'follow',
  'favorite'
]

history = {
  'action': {},
  'finish': {},
  'stay': {},
  'neg': {},
  'unfinish': {},
  'unstay': {},
  'latest': {}
}

for action in ACTIONS:
  history[action] = {}

userids = [int(x.strip().split()[0]) for x in open('../input/user_vocab.txt').readlines()]
for userid in userids:
  history['action'][userid] = []
  history['finish'][userid] = []
  history['stay'][userid] = []
  history['neg'][userid] = []
  history['unfinish'][userid] = []
  history['unstay'][userid] = []
  history['latest'][userid] = []
  for action in ACTIONS:
    history[action][userid] = []

def is_neg(row, play=None):
  for action in ACTIONS:
    if row[action] > 0:
      return False

  return True


d = d.sort_values(['date_'], ascending=False)
for _, row in tqdm(d.iterrows(), total=len(d), desc='user_action'):
  feedid = int(row['feedid'])
  userid = int(row['userid'])
  day = int(row['date_'])
  feedid = (feedid, day)
  history['latest'][userid].append(feedid)
  if row['finish_rate'] > 0.99:
    history['finish'][userid].append(feedid)
  if row['finish_rate'] < 0.01:
    history['unfinish'][userid].append(feedid)
  if row['stay_rate'] > 1:
    history['stay'][userid].append(feedid)
  if row['stay_rate'] < 0.01:
    history['unstay'][userid].append(feedid)
  is_neg_row = is_neg(row)
  if not is_neg_row:
    history['action'][userid].append(feedid)
  else:
    history['neg'][userid].append(feedid)
  for action in ACTIONS:
    if row[action] > 0:
      history[action][userid].append(feedid)


# In[5]:


gezi.save_pickle(history, '../input/history2.pkl')


# In[6]:


history['read_comment'][131440]


# In[7]:


m = {
  'userid': [],
  'action': [],
  'action_len': [],
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
  'latest': [],
  'latest_len': [],
}
OTHER_ACTIONS = ['action', 'neg', 'finish', 'unfinish', 'stay', 'unstay', 'latest']
for action in ACTIONS:
  m[action] = []
  m[f'{action}_len'] = []

for userid in tqdm(userids):
  m['userid'].append(userid)
  for action in ACTIONS + OTHER_ACTIONS:
    m[action].append(' '.join(map(str, history[action][userid])))
    m[f'{action}_len'].append(len(history[action][userid]))


# In[8]:


his_df = pd.DataFrame(m)


# In[9]:


his_df.head()


# In[10]:


his_df.describe(percentiles=[.25,.5,.75,.9,.95,.99, .999])


# In[11]:


his_df.to_csv('../input/history2.csv', index=False)


# In[ ]:




