#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict
import glob
import sys,os
sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')
import gezi
from gezi import tqdm
tqdm.pandas()


# In[39]:


with gezi.Timer('read train user_actions'):
  d = pd.read_csv('../input/user_action.csv')
  try:
    d1 = pd.read_csv('../input/v1/user_action.csv')
    d = pd.concat([d, d1])
  except Exception:
    pass


# In[40]:


with gezi.Timer('read test and merge'):
  dt = pd.read_csv('../input/test_a.csv')
  try:
    dt1a = pd.read_csv('../input/v1/test_a.csv')
    dt1b = pd.read_csv('../input/v1/test_b.csv')
    dt = pd.concat([dt, dt1a, dt1b])
  except Exception:
    dtb = pd.read_csv('../input/test_b.csv')
    dt = pd.concat([dt, dtb])


# In[41]:


dt['date_'] = 15


# In[42]:


set(d.date_)


# In[43]:


dshow = pd.concat([d[['userid', 'feedid', 'date_']],
                  dt[['userid', 'feedid', 'date_']]])
dshow = dshow.sort_values(['date_'], ascending=True)


# In[44]:


days = [
  # # 包含14号 1表示half 
  # 并且14号只有一半用户使用 userid % 2 == 0 14.5 (也许13.5更合适) date2<=13.5
  # 也就是数据量 13 < 14.5(13.5) < 14 < 15
  (14, 1),  # valid show一半用户可见
  (13, 0),  # 截止到13号数据,valid show完全不可见
  (14, 0),  # 包含14号全部用户数据 (valid show完全可见)
  (15, 0)   # 包含15号(test_a)全部数据
]


# In[45]:


feeds = {}
def cache_feed():
  df = pd.read_csv('../input/feed_info.csv')
  df = df.fillna(-1)
  for row in tqdm(df.itertuples(), total=len(df), desc='feed_info'):
    row = row._asdict()               
    feeds[row['feedid']] = row
cache_feed()


# In[1]:


MAX_LEN = 1000


# In[ ]:


MODE = 'ignore' # or 'split'


# In[ ]:


def print_items(items, f, max_len=None, mode='ignore'):
  l = []
  if not max_len:
    l = [items]
  else:
    len_ = len(items)
    if len_ > max_len and mode == 'ignore':
      return
    count = -(-len_ // max_len)
    l = np.array_split(items, count)
    if count > 1:
      l[-1] = items[-max_len:]
  for items in l:
    print(' '.join(map(str, items)), file=f)


# In[53]:


def write_user_corpus(d, mark, group_byday=False):
  # TODO doc 对应的user比较多 特别复赛20w user， 初赛只有2w 可能需要shuffle 大的doc 然后限定小窗口
  keys = ['feedid'] if not group_byday else ['feedid', 'date_']
  users = d.groupby(keys)['userid'].progress_apply(list).reset_index(name='users')
  users['num_users'] = users.users.apply(len)
  corpus_name = 'user_corpus' if not group_byday else 'user_day_corpus'
  with open(f'../input/{mark}/{corpus_name}.txt', 'w') as f:
    for users_ in tqdm(users.users, desc=f'{mark}/{corpus_name}'):
      # TODO 如果过长 一个doc展现了非常多的user 是忽略比较好 还是分段比较好
      print_items(users_, f, MAX_LEN)


# In[54]:


def write_doc_corpus(d, mark, group_byday=False):
  # TODO doc 对应的user比较多 特别复赛20w user， 初赛只有2w 可能需要shuffle 大的doc 然后限定小窗口
  keys = ['userid'] if not group_byday else ['feedid', 'date_']
  docs = d.groupby(keys)['feedid'].progress_apply(list).reset_index(name='docs')
  docs['num_docs'] = docs.docs.apply(len)
  corpus_name = 'doc_corpus' if not group_byday else 'doc_day_corpus'
  with open(f'../input/{mark}/{corpus_name}.txt', 'w') as f:
    for docs_ in tqdm(docs.docs, desc=f'{mark}/{corpus_name}'):
      # TODO doc 虽然能跑 但是非常慢 大约需要6-7个小时 可以考虑也忽略掉长的? 一个user有过多doc的
      print_items(docs_, f)
  attrs = ['author', 'singer', 'song']
  names = ['authorid', 'bgm_singer_id', 'bgm_song_id']
  for attr, name in zip(attrs, names):
    corpus_name = f'{attr}_corpus' if not group_byday else f'{attr}_day_corpus'
    with open(f'../input/{mark}/{corpus_name}.txt', 'w') as f:
      for docs_ in tqdm(docs.docs, desc=f'{mark}/{corpus_name}'):
        items = [feeds[doc][name] for doc in docs_]
        print_items(items, f)


# In[55]:


t = tqdm(days, total=len(days))
for day, is_half in t:
  mark = day if not is_half else day + 0.5
  os.system(f'mkdir -p ../input/{mark}')
  if not is_half:
    d = dshow[dshow.date_ <= day]
  else:
    d = dshow[(dshow.date_ < day) | (dshow.date_ == day) & (dshow.userid % 2 == 0)]
  print({'day': mark, 'total': len(d)})
  write_user_corpus(d, mark, group_byday=False)
  write_doc_corpus(d, mark, group_byday=False)
  write_user_corpus(d, mark, group_byday=True)
  write_doc_corpus(d, mark, group_byday=True)


# In[ ]:


# 暂时留记录 但是不生成tag key corpu


# In[34]:


# with open('../input/tag_test_corpus.txt', 'w') as f:
#   for docs_ in tqdm(docs.docs):
#     tags = [str(feeds[doc]['manual_tag_list']).replace(';', ' ') for doc in docs_]
#     tags = set(' '.join(tags).split())
#     print(' '.join(tags), file=f)
#     tags2 = [' '.join([x.split()[0] for x in str(feeds[doc]['machine_tag_list']).split(';')]) for doc in docs_]
#     tags2 = set(' '.join(tags2).split())
#     print(' '.join(tags2), file=f)


# In[35]:


# with open('../input/key_test_corpus.txt', 'w') as f:
#   for docs_ in tqdm(docs.docs):
#     tags = [str(feeds[doc]['manual_keyword_list']).replace(';', ' ') for doc in docs_]
#     tags = set(' '.join(tags).split())
#     print(' '.join(tags), file=f)
#     tags2 = [' '.join([x.split()[0] for x in str(feeds[doc]['machine_keyword_list']).split(';')]) for doc in docs_]
#     tags2 = set(' '.join(tags2).split())
#     print(' '.join(tags2), file=f)


# In[ ]:





# In[ ]:




