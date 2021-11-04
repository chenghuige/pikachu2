#!/usr/bin/env python
# coding: utf-8

# In[10]:


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


# In[11]:


DYNAMIC_LEN = False
PARALLEL = True
# if 'tione' in os.environ['PATH']:
#   PARALLEL = False


# In[12]:


with gezi.Timer('read user_action2.feather', True):
  d = pd.read_feather('../input/user_action2.feather')
  d.feedid = d.feedid.astype(int)
  d.userid = d.userid.astype(int)
  d.date_ = d.date_.astype(int)


# In[13]:


d.head()


# In[14]:


with gezi.Timer('read history.pkl', True):
  history = gezi.read_pickle('../input/history.pkl')


# In[15]:


if not DYNAMIC_LEN:
  HIST_LENS = {
    "read_comment": 50,  # 是否查看评论
    "like": 50,  # 是否点赞
    "click_avatar": 30,  # 是否点击头像
    "forward": 20,  # 是否转发
    "favorite": 10,  # 是否收藏
    "comment": 3,  # 是否发表评论
    "follow": 5,  # 是否关注
    "pos": 50,
    "neg": 50,
    "finish": 50,
    "unfinish": 50,
    'stay': 50,
    'unstay': 20,
  #   'unfinish2': 50,
  #   'unstay2': 50,
  #   "latest": 100, #  latest 包括不包括当天的show 对应所有都用户交互信息
  #   "today": 100, # 当天的show
    # "show": 100, # 不包括当天的show 相当于 action | neg
  }
else:
  HIST_LENS = {
    "read_comment": 200,  # 是否查看评论
    "like": 200,  # 是否点赞
    "click_avatar": 200,  # 是否点击头像
    "forward": 200,  # 是否转发
    "favorite": 200,  # 是否收藏
    "comment": 200,  # 是否发表评论
    "follow": 200,  # 是否关注
    "pos": 50,
    "neg": 50,
    "finish": 50,
    "unfinish": 50,
    'stay': 50,
    'unstay': 20,
  #   'unfinish2': 50,
  #   'unstay2': 50,
  #   "latest": 500, #  latest 包括不包括当天的show 对应所有都用户交互信息
  #   "today": 100, # 当天的show
  }
EMPTY_ID = 1
HIS_ACTIONS = list(list(history.values())[0].keys())


# In[16]:


vocab = gezi.Vocab('../input/doc_vocab.txt')


# In[17]:


def get_history_day_(userid, action, day):  
  feeds, days = [], []
  if day > 1:
    hist = history[userid][action]
    for feedid_, day_ in hist:
      if day_ < day:
        feeds.append(vocab.id(feedid_))
        days.append(day_)
  return feeds, [day - x + 1 for x in days]

def get_history_day(userids, day):
  history_day = {}
  for userid in tqdm(userids):
    if day not in dates_[userid]:
      continue
    history_day[userid] = {}
    for action in HIS_ACTIONS:
      history_day[userid][action] = get_history_day_(userid, action, day)
  return history_day


# In[18]:


dates = d.groupby(['userid'])['date_'].progress_apply(set).reset_index(name='dates')


# In[19]:


# dates.head()


# In[20]:


# let all userid have day 15 to lookup history for unkonw test_b
dates.dates = dates.dates.apply(lambda x: list(x) + [15])


# In[21]:


# dates.head()


# In[22]:


dates_ = {}
for i in tqdm(range(len(dates))):
  dates_[dates.userid.values[i]] = set(dates.dates.values[i])


# In[23]:


# dates_


# In[24]:


# len(dates_)


# In[25]:


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


# In[26]:


history_day = gezi.read_pickle('../input/history_14.pkl')


# In[27]:


for action in HIS_ACTIONS:
  uid = list(history_day.keys())[0]
  print(action, history_day[uid][action])


# In[28]:


history_day = gezi.read_pickle('../input/history_14.pkl')
for uid in history_day:
  if len(history_day[uid]['pos'][0]) == 0:
    print(uid, history_day[uid])
    break
print(history_day[100110])


# In[29]:


# history[1]


# In[ ]:





# In[ ]:





# In[ ]:




