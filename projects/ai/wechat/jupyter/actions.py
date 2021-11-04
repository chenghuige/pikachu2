#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from collections import defaultdict
import glob
import sys 
sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')
import gezi
from gezi import tqdm


# In[2]:


ACTIONS = [
  'read_comment',
  'like',
  'click_avatar',
  'forward',
  'favorite',
  'comment',
  'follow'
]


# In[3]:


feeds = {}
def cache_feed():
  df = pd.read_csv('../input/feed_info.csv')
  df = df.fillna(-1)
  for row in tqdm(df.itertuples(), total=len(df), desc='feed_info'):
    row = row._asdict()
    feeds[row['feedid']] = row
cache_feed()


# In[4]:


def num_actions(row):
  cnt = 0
  for action in ACTIONS:
    if row[action] > 0:
      cnt += 1
  return cnt


# In[5]:


with gezi.Timer('read_actions csv', True):
  d = pd.read_csv('../input/user_action.csv')
  d['version'] = 2
  try:
    d1 = pd.read_csv('../input/v1/user_action.csv')
    d1['version'] = 1
    d = pd.concat([d, d1])
  except Exception:
    pass
  d.feedid = d.feedid.astype(int)
  d.userid = d.userid.astype(int)
  d.date_ = d.date_.astype(int)


# In[6]:


d.head()


# In[7]:


finish_rates = []
stay_rates = []
is_firsts = []
actions = []

d = d.sort_values(['date_'], ascending=True)
m = defaultdict(int)
for row in tqdm(d.itertuples(), total=len(d), desc='user_action'):
  row = row._asdict()
  vtime = min(feeds[row['feedid']]['videoplayseconds'], 60) * 1000 # to ms
  vtime = feeds[row['feedid']]['videoplayseconds'] * 1000 # to ms
  finish_rates.append(row['play'] / vtime)
  stay_rates.append(row['stay'] / vtime)
  userid = row['userid']
  feedid = row['feedid']
  key = f'{userid}_{feedid}'
  m[key] += 1
  is_firsts.append(int(m[key] == 1))
  actions.append(num_actions(row))
d['finish_rate'] = finish_rates
d['stay_rate'] = stay_rates
d['is_first'] = is_firsts
d['actions'] = actions


# In[8]:


# d.head()


# In[9]:


# d.describe()


# In[10]:


# timer = gezi.Timer('save user_action2.csv')
# d.to_csv('../input/user_action2.csv', index=False)
# timer.print_elapsed()


# In[14]:


# timer = gezi.Timer('read user_action2.csv', True)
# d = pd.read_csv('../input/user_action2.csv')
# d.feedid = d.feedid.astype(int)
# d.userid = d.userid.astype(int)
# d.date_ = d.date_.astype(int)
# timer.print_elapsed()
timer = gezi.Timer('save user_action2.feather', True)
d.reset_index().to_feather('../input/user_action2.feather')
timer.print()


# In[15]:


timer = gezi.Timer('read user_action2.feather', True)
pd.read_feather('../input/user_action2.feather')
timer.print()


# In[17]:


for i in tqdm(range(14)):
  day = i + 1
  d[d.date_ == day].reset_index().to_feather(f'../input/user_action2_{day}.feather')


# In[ ]:





# In[ ]:




