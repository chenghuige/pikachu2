#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
tqdm.pandas()


# In[ ]:


ACTIONS = ["read_comment", "like", "click_avatar",  "forward", "comment", "follow", "favorite"]
ACTIONS2 = ACTIONS + ['finish', 'stay']
HIS_ACTIONS = ACTIONS2 + [
    'pos', 'neg', 'unfinish', 'unstay'
]


# In[ ]:


feeds = {}
def cache_feed():
  df = pd.read_csv('../input/feed_info.csv')
  df = df.fillna(-1)
  for row in tqdm(df.itertuples(), total=len(df), desc='feed_info'):
    row = row._asdict()
    feeds[row['feedid']] = row
cache_feed()


# In[ ]:


with gezi.Timer('read train user_actions'):
  d = pd.read_feather('../input/user_action2.feather')
with gezi.Timer('read test and merge'):
  dt = pd.read_csv('../input/test_a.csv')
  try:
    dt1a = pd.read_csv('../input/v1/test_a.csv')
    dt1b = pd.read_csv('../input/v1/test_b.csv')
    dt = pd.concat([dt, dt1a, dt1b])
  except Exception:
    dtb = pd.read_csv('../input/test_b.csv')
    dt = pd.concat([dt, dtb])
dt['date_'] = 15
dshow = pd.concat([d[['userid', 'feedid', 'date_']],
                  dt[['userid', 'feedid', 'date_']]])
dshow = dshow.sort_values(['date_'], ascending=True)
d = d.sort_values(['date_'], ascending=True)


# In[ ]:


dshow[dshow.date_==15].head()


# In[ ]:


# tag_info = {}
# for row in tqdm(fd.itertuples(), total=len(fd), desc='feed_info-manual_tag'):
#   row = row._asdict()
#   manual_tags = str(row['manual_tag_list']).split(';')
#   tag_info[int(row['feedid'])] = manual_tags


# In[ ]:


d.head()


# In[ ]:


df = d


# In[ ]:


def write_corpus(day):
  d = df[df.date_ < day]
  dfs = {}
  t = tqdm(ACTIONS)
  for action in t:
    t.set_postfix({'action': action})
    dfs[action] = d[d[action] == 1].groupby(['userid'])['feedid'].progress_apply(list).reset_index(name=action)
  dfs['finish'] = d[d.finish_rate > 0.99].groupby(['userid'])['feedid'].progress_apply(list).reset_index(name='finish')
  dfs['unfinish'] = d[d.finish_rate < 0.01].groupby(['userid'])['feedid'].progress_apply(list).reset_index(name='unfinish')
  dfs['stay'] = d[d.stay_rate > 1].groupby(['userid'])['feedid'].progress_apply(list).reset_index(name='stay')
  dfs['unstay'] = d[d.stay_rate < 0.01].groupby(['userid'])['feedid'].progress_apply(list).reset_index(name='unstay')
  dfs['neg'] = d[d.actions == 0].groupby(['userid'])['feedid'].progress_apply(list).reset_index(name='neg')
  dfs['pos'] = d[d.actions > 0].groupby(['userid'])['feedid'].progress_apply(list).reset_index(name='pos')
  # dfs['show'] = dshow.groupby(['userid'])['feedid'].progress_apply(list).reset_index(name='show')
  for action in tqdm(dfs):
    with open(f'../input/{day}/doc_{action}.txt', 'w') as f:
      for l in dfs[action][action].values:
        print(' '.join(map(str, l)), file=f)
    keys = ['author', 'singer', 'song']
    keys_ = ['authorid', 'bgm_singer_id', 'bgm_song_id']
    for key, key_ in zip(keys, keys_):
      with open(f'../input/{day}/{key}_{action}.txt', 'w') as f:
        for l in dfs[action][action].values:
          print(' '.join(map(lambda x:str(feeds[x][key_]), l)), file=f)
  return dfs


# In[ ]:


dfs_14 = write_corpus(14)


# In[ ]:


dfs_15 = write_corpus(15)


# In[ ]:


# dfs['read_comment'].merge(dfs['comment'], on='userid')


# In[ ]:




