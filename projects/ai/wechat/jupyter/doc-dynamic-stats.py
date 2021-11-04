#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict, ChainMap, Counter
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
from gezi import tqdm
tqdm.pandas()


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
DAYS = 15


# In[3]:


timer = gezi.Timer('read user_action2.feather', True)
d = pd.read_feather('../input/user_action2.feather')
d.feedid = d.feedid.astype(int)
d.userid = d.userid.astype(int)
d.date_ = d.date_.astype(int)
timer.print()


# In[4]:


all_feedids = set(pd.read_csv('../input/feed_info.csv').feedid)


# In[5]:


d = d.sort_values(['date_'], ascending=True)


# In[6]:


d.head()


# In[7]:


d.read_comment.mean()


# In[8]:


doc_dynamic_feature = {}
for feedid in tqdm(all_feedids):
  doc_dynamic_feature[int(feedid)] = {}


# In[9]:


dates = d.groupby(['feedid'])['date_'].progress_apply(list).reset_index(name='dates')


# In[10]:


dates['dates'] = dates.dates.apply(lambda x:dict(Counter(x)))


# In[11]:


dates.head()


# In[12]:


days = DAYS
for feedid in all_feedids:
  shows = [0] * (days + 1)
  doc_dynamic_feature[feedid]['shows'] = shows
  
for row in tqdm(dates.itertuples(), total=len(dates), desc='shows'):
  row = row._asdict()
  dates_ = row['dates']
  shows = [0] * (days + 1)
  for i in range(days):
    i += 1
    if i in dates_:
      shows[i] = dates_[i]
  doc_dynamic_feature[int(row['feedid'])]['shows'] = shows


# In[13]:


doc_dynamic_feature[d.feedid.values[0]]


# In[14]:


def gen_doc_dynamic(d, feedids=None):
  days = DAYS
  if feedids is not None:
    d = d[d.feedid.isin(set(feedids))]
  else:
    feedids = set(d.feedid)
  dg = d.groupby(['feedid', 'date_'])
  actions = ACTIONS + ['actions', 'finish_rate', 'stay_rate']
  doc_dynamic_feature = {}
  
  for feedid in feedids:
    doc_dynamic_feature[int(feedid)] = {}
    for action in actions:
      doc_dynamic_feature[int(feedid)][action] = [0] * (days + 1)

  t = tqdm(actions)
  for action in t:
#   for action in actions:
#     t.set_postfix({'action': action})
    da = dg[action].progress_apply(sum).reset_index(name=f'{action}_count')
#     for row in tqdm(da.itertuples(), total=len(da), desc=f'{action}_count'):
    for row in da.itertuples():
      row = row._asdict()
      date = row['date_']
      feedid = int(row['feedid'])
      ddf = doc_dynamic_feature[int(row['feedid'])]
      ddf[action][date] = row[f'{action}_count']
  return doc_dynamic_feature


# In[15]:


# gen_doc_dynamic(d)


# In[16]:


import pymp
nw = cpu_count()
feedids_list = np.array_split(list(all_feedids), nw)
res = Manager().dict()
with pymp.Parallel(nw) as p:
  for i in p.range(nw):
    res[i] = gen_doc_dynamic(d, feedids_list[i])
doc_dynamic_feature2 = dict(ChainMap(*res.values()))

# pfunc = functools.partial(gen_doc_dynamic, d=d)
# with Pool(nw) as p:
#   res = p.map(pfunc, feedids_list)
# doc_dynamic_feature2 = dict(ChainMap(*res))


# In[17]:


print(len(doc_dynamic_feature), len(doc_dynamic_feature2))


# In[18]:


for feedid in doc_dynamic_feature:
  doc_dynamic_feature[feedid].update(doc_dynamic_feature2[feedid])


# In[19]:


doc_dynamic_feature[d.feedid.values[0]]


# In[20]:


dates[dates.feedid==36523]


# In[21]:


doc_dynamic_feature[36523]


# In[22]:


gezi.save_pickle(doc_dynamic_feature, '../input/doc_dynamic_feature.pkl')


# In[23]:


d.groupby(['date_'])['userid'].count()


# In[ ]:





# In[ ]:




