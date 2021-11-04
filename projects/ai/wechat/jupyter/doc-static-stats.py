#!/usr/bin/env python
# coding: utf-8

# In[4]:


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


# In[5]:


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


# In[6]:


timer = gezi.Timer('read user_action2.feather', True)
d = pd.read_feather('../input/user_action2.feather')
d.feedid = d.feedid.astype(int)
d.userid = d.userid.astype(int)
d.date_ = d.date_.astype(int)
timer.print()


# In[7]:


d = d.sort_values(['date_'], ascending=True)


# In[8]:


d.head()


# In[9]:


d.read_comment.mean()


# In[10]:


action_default_vals = {}
for action in ['actions', 'finish_rate', 'stay_rate'] + ACTIONS:
  action_default_vals[action] = d[action].mean()
gezi.save_pickle(action_default_vals, '../input/action_default_vals.pkl')


# In[7]:


dates = d.groupby(['feedid'])['date_'].progress_apply(list).reset_index(name='dates')


# In[8]:


dates['dates'] = dates.dates.apply(lambda x:dict(Counter(x)))


# In[9]:


dates


# In[10]:


dates['start_day'] = dates.dates.apply(lambda x: min(x.keys()))


# In[1]:


dates.head()


# In[12]:


dates.to_csv('../input/doc_static_feature.csv', index=False)


# In[ ]:




