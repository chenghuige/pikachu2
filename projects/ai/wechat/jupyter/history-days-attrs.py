#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict, ChainMap
import glob
import sys, os
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


DYNAMIC_LEN = False
PARALLEL = True
if 'tione' in os.environ['PATH']:
  PARALLEL = False


# In[5]:


with gezi.Timer('read user_action2.feather', True):
  d = pd.read_feather('../input/user_action2.feather')
  d.feedid = d.feedid.astype(int)
  d.userid = d.userid.astype(int)
  d.date_ = d.date_.astype(int)


# In[6]:


d.head()


# In[7]:


with gezi.Timer('read history.pkl', True):
  history = gezi.read_pickle('../input/history.pkl')


# In[9]:


vocab_names = [
                'user', 'doc',
                'author', 'singer', 'song',
                'key', 'tag', 'word', 'char'
              ]
vocabs = {}

for vocab_name in vocab_names:
  vocab_file =  f'../input/{vocab_name}_vocab.txt'
  vocab = gezi.Vocab(vocab_file)
  vocabs[vocab_name] = vocab


# In[ ]:


def get_history_day_(userid, action, day):  
  feeds = []
  if day > 1:
    hist = history[userid][action]
    for feedid_, day_ in hist:
      if day_ < day:
        

def get_history_day(userids, day):
  history_day = {}
  for userid in tqdm(userids):
    if day not in dates_[userid]:
      continue
    history_day[userid] = {}
    for action in HIS_ACTIONS:
      history_day[userid][action] = get_history_day_(userid, action, day)
  return history_day


# In[ ]:


userids = list(history.keys())


# In[ ]:


dates = d.groupby(['userid'])['date_'].progress_apply(set).reset_index(name='dates')


# In[ ]:


# dates.head()


# In[ ]:


# let all userid have day 15 to lookup history for unkonw test_b
dates.dates = dates.dates.apply(lambda x: list(x) + [15])


# In[ ]:


# dates.head()


# In[ ]:


dates_ = {}
for i in tqdm(range(len(dates))):
  dates_[dates.userid.values[i]] = set(dates.dates.values[i])


# In[ ]:


# dates_


# In[ ]:


# len(dates_)


# In[ ]:


DAYS = 15
nw = cpu_count()
userids = list(history.keys())
userids_list = np.array_split(userids, nw)

## though write easy, but might connect fail if multiprocessing need long time run
## not as stable as Pool, only could be used for not too heavy ..
## might be call multiple times of with pymp.Parallel(nw) as p, or too much time waiting
## for writting file
##   File "/home/tione/notebook/envs/pikachu/lib/python3.6/multiprocessing/connection.py", line 614, in SocketClient
##     s.connect(address)
## ConnectionRefusedError: [Errno 111] Connection refused

## 这些都会hang 后面 可能由于 单个进程等待时间过长
# for day in tqdm(range(DAYS)):
#   day += 1
#   res = Manager().dict()
#   with pymp.Parallel(nw) as p:
#     for i in p.range(nw):
#       res[i] = get_history_day(userids_list[i], day)
#   history_day = dict(ChainMap(*res.values()))
#   gezi.save_pickle(history_day, f'../input/history_{day}.pkl')

# for day in tqdm(range(DAYS)):
#   day += 1
# #   if day == 1:
# #     continue
#   pfunc = functools.partial(get_history_day, day=day)
#   with Pool(nw) as p:
#     res = p.map(pfunc, userids_list)
#   history_day = dict(ChainMap(*res))
#   gezi.save_pickle(history_day, f'../input/history_{day}.pkl')

if not PARALLEL:
  for day in tqdm(range(DAYS)):
    day += 1
    history_day = get_history_day(userids, day)
    gezi.save_pickle(history_day, f'../input/history_{day}.pkl')
else:
  # 如果没有OOM 超过内存 那么pymp 一个处理一个 一一对应是没问题的 cpu机器没问题
  # 如果OOM 比如设置nw=8 那么下面就会有问题.. 启用进程数最好还是和p.range一致
  # 为了安全 还是不走多进程
  nw = min(nw, DAYS)
  with pymp.Parallel(nw) as p:
    for day in p.range(DAYS):
      day += 1
      history_day = get_history_day(userids, day)
      gezi.save_pickle(history_day, f'../input/history_{day}.pkl')


# In[ ]:


history_day = gezi.read_pickle('../input/history_14.pkl')


# In[ ]:


for action in HIS_ACTIONS:
  uid = list(history_day.keys())[0]
  print(action, history_day[uid][action])


# In[ ]:


history_day = gezi.read_pickle('../input/history_14.pkl')
for uid in history_day:
  if len(history_day[uid]['pos'][0]) == 0:
    print(uid, history_day[uid])
    break
print(history_day[100110])


# In[ ]:


# history[1]


# In[ ]:





# In[ ]:





# In[ ]:




