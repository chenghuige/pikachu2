#!/usr/bin/env python
# coding: utf-8

# In[229]:


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


# In[230]:


INCLUDE_TODAY = True


# In[231]:


MAX_VALID_DAY = 14 if INCLUDE_TODAY else 13
MAX_TEST_DAY = 15 if INCLUDE_TODAY else 14


# In[232]:


MAX_VALID_DAY = 13
MAX_TEST_DAY = 15


# In[233]:


d = pd.read_csv('../input/feed_info.csv')


# In[234]:


d.head()


# In[235]:


MAX_TAGS = 14
MAX_KEYS = 18
DESC_LEN = 128
DESC_CHAR_LEN = 256
WORD_LEN = 256
START_ID = 3
UNK_ID = 1
EMPTY_ID = 2
NAN_ID = -1


# In[236]:


d = d.fillna(NAN_ID)


# In[237]:


d.feedid = d.feedid.astype(int)
d.authorid = d.authorid.astype(int)
d.bgm_singer_id = d.bgm_singer_id.astype(int)
d.bgm_song_id = d.bgm_song_id.astype(int)
with open('../input/word_corpus.txt', 'w') as f:
  for i, desc in enumerate(d.description.values):
    if str(desc) == str(NAN_ID):
      continue
    print(desc, file=f)
  for i, ocr in enumerate(d.ocr.values):
    if str(ocr) == str(NAN_ID):
      continue
    print(ocr, file=f)
  for i, asr in enumerate(d.asr.values):
    if str(asr) == str(NAN_ID):
      continue
    print(asr, file=f)
    
with open('../input/char_corpus.txt', 'w') as f:
  for i, desc in enumerate(d.description_char.values):
    if str(desc) == str(NAN_ID):
      continue
    print(desc, file=f)
  for i, ocr in enumerate(d.ocr_char.values):
    if str(ocr) == str(NAN_ID):
      continue
    print(ocr, file=f)
  for i, asr in enumerate(d.asr_char.values):
    if str(asr) == str(NAN_ID):
      continue
    print(asr, file=f)


# In[238]:


with open('../input/author_corpus.txt', 'w') as f:
  for i, author in enumerate(d.authorid.values):
    if author == NAN_ID:
      continue
    print(author, file=f)
with open('../input/singer_corpus.txt', 'w') as f:
  for i, singer in enumerate(d.bgm_singer_id.values):
    if singer == NAN_ID:
      continue
    print(singer, file=f)
with open('../input/song_corpus.txt', 'w') as f:
  for i, song in enumerate(d.bgm_song_id.values):
    if song == NAN_ID:
      continue
    print(song, file=f)


# In[239]:


with open('../input/tag_corpus.txt', 'w') as f:
  for i, tag in enumerate(d.manual_tag_list):
    if str(tag) == str(NAN_ID):
      continue
    else:
      print(str(tag).replace(';', ' '), file=f)
  for i, tag in enumerate(d.machine_tag_list):
    if str(tag) == str(NAN_ID):
      continue
    else:
      print(' '.join([x.split()[0] for x in tag.split(';')]), file=f)
    
with open('../input/key_corpus.txt', 'w') as f:
  for i, kw in enumerate(d.manual_keyword_list):
    if str(kw) == str(NAN_ID):
      continue
    else:
      print(str(kw).replace(';', ' '), file=f)
  for kw in d.machine_keyword_list:
    if str(kw) == str(NAN_ID):
      continue
    else:
      print(str(kw).replace(';', ' '), file=f)


# In[240]:


d2 = pd.read_csv('../input/user_action2.csv')


# In[241]:


dt = pd.read_csv('../input/test_a.csv')


# In[242]:


dt2 = pd.read_csv('../input/test_b.csv')


# In[243]:


dt2.head()


# In[244]:


dt['date_'] = 15
dt2['date_'] = 15


# In[245]:


set(d2.date_)


# In[246]:


# dshows = d2[['userid', 'feedid', 'date_']]
dshows = pd.concat([d2[['userid', 'feedid', 'date_']],
                  dt[['userid', 'feedid', 'date_']], 
                  dt2[['userid', 'feedid', 'date_']]])


# In[247]:


# TODO add test


# In[248]:


dshows.head()


# In[249]:


tqdm.pandas()


# In[250]:


# # 似乎不考虑date_效果更好 这个depreciate
# docs = dshows.groupby(['userid', 'date_'])['feedid'].progress_apply(list).reset_index(name='docs')
# docs.head()


# In[22]:


# with open('../input/doc_corpus.txt', 'w') as f:
#   for docs_ in docs.docs:
#     print(' '.join(map(str, docs_)), file=f)


# In[23]:


# docs['num_docs'] = docs.docs.apply(len)


# In[24]:


# docs.num_docs.describe()


# In[251]:


docs = dshows[dshows.date_ <= MAX_TEST_DAY].groupby(['userid'])['feedid'].progress_apply(list).reset_index(name='docs')
docs.head()


# In[252]:


docs_test = docs


# In[253]:


# 似乎不考虑date_效果更好，好很多
with open('../input/doc_test_corpus.txt', 'w') as f:
  for docs_ in docs.docs:
    print(' '.join(map(str, docs_)), file=f)


# In[254]:


docs['num_docs'] = docs.docs.apply(len)


# In[255]:


docs.num_docs.describe()


# In[256]:


docs = dshows[dshows.date_ <= MAX_VALID_DAY].groupby(['userid'])['feedid'].progress_apply(list).reset_index(name='docs')
docs.head()


# In[257]:


docs_valid = docs


# In[258]:


with open('../input/doc_valid_corpus.txt', 'w') as f:
  for docs_ in docs.docs:
    print(' '.join(map(str, docs_)), file=f)


# In[259]:


docs['num_docs'] = docs.docs.apply(len)


# In[260]:


docs.num_docs.describe()


# In[261]:


users = d2[(d2.date_ < MAX_TEST_DAY) & (d2.actions > 0)].groupby(['feedid'])['userid'].progress_apply(list).reset_index(name='users')


# In[262]:


users.head()


# In[263]:


with open('../input/user_action_test_corpus.txt', 'w') as f:
  for users_ in users.users:
    print(' '.join(map(str, users_)), file=f)


# In[264]:


users['num_users'] = users.users.apply(len)


# In[265]:


users.num_users.describe()


# In[266]:


users = d2[(d2.date_ < MAX_VALID_DAY) & (d2.actions > 0)].groupby(['feedid'])['userid'].progress_apply(list).reset_index(name='users')


# In[267]:


with open('../input/user_action_valid_corpus.txt', 'w') as f:
  for users_ in users.users:
    print(' '.join(map(str, users_)), file=f)


# In[268]:


users['num_users'] = users.users.apply(len)


# In[269]:


users.num_users.describe()


# In[270]:


# TODO doc 对应的user比较多 特别复赛20w user， 初赛只有2w 可能需要shuffle 大的doc 然后限定小窗口
users = dshows[dshows.date_ <= MAX_TEST_DAY].groupby(['feedid', 'date_'])['userid'].progress_apply(list).reset_index(name='users')


# In[271]:


with open('../input/user_test_corpus.txt', 'w') as f:
  for users_ in users.users:
    print(' '.join(map(str, users_)), file=f)


# In[272]:


users['num_users'] = users.users.apply(len)


# In[273]:


users.num_users.describe()


# In[274]:


users = dshows[dshows.date_ <= MAX_VALID_DAY].groupby(['feedid', 'date_'])['userid'].progress_apply(list).reset_index(name='users')


# In[275]:


with open('../input/user_valid_corpus.txt', 'w') as f:
  for users_ in users.users:
    print(' '.join(map(str, users_)), file=f)


# In[276]:


users['num_users'] = users.users.apply(len)


# In[277]:


users.num_users.describe()


# In[278]:


feeds = {}
def cache_feed():
  df = pd.read_csv('../input/feed_info.csv')
  df = df.fillna(-1)
  for row in tqdm(df.itertuples(), total=len(df), ascii=True, desc='feed_info'):
    row = row._asdict()               
    feeds[row['feedid']] = row
cache_feed()


# In[279]:


docs_test.num_docs.max()


# In[280]:


docs_valid.num_docs.max()


# In[281]:


# 下面对应的都是valid
docs = docs_valid
with open('../input/author_valid_corpus.txt', 'w') as f:
  for docs_ in tqdm(docs.docs):
    authors = [feeds[doc]['authorid'] for doc in docs_]
    print(' '.join(map(str, authors)), file=f)


# In[282]:


with open('../input/singer_valid_corpus.txt', 'w') as f:
  for docs_ in tqdm(docs.docs):
    singers = [feeds[doc]['bgm_singer_id'] for doc in docs_]
    print(' '.join(map(str, singers)), file=f)


# In[283]:


with open('../input/song_valid_corpus.txt', 'w') as f:
  for docs_ in tqdm(docs.docs):
    songs = [feeds[doc]['bgm_song_id'] for doc in docs_]
    print(' '.join(map(str, songs)), file=f)


# In[284]:


# with open('../input/tag_valid_corpus.txt', 'w') as f:
#   for docs_ in tqdm(docs.docs):
#     tags = [str(feeds[doc]['manual_tag_list']).replace(';', ' ') for doc in docs_]
#     print(' '.join(map(str, tags)), file=f)
#     tags2 = [' '.join([x.split()[0] for x in str(feeds[doc]['machine_tag_list']).split(';')]) for doc in docs_]
#     print(' '.join(map(str, tags2)), file=f)


# In[285]:


# with open('../input/key_valid_corpus.txt', 'w') as f:
#   for docs_ in tqdm(docs.docs):
#     keys = [str(feeds[doc]['manual_keyword_list']).replace(';', ' ') for doc in docs_]
#     print(' '.join(map(str, keys)), file=f)
#     keys2 = [' '.join([x.split()[0] for x in str(feeds[doc]['machine_keyword_list']).split(';')]) for doc in docs_]
#     print(' '.join(map(str, keys2)), file=f)


# In[286]:


with open('../input/tag_valid_corpus.txt', 'w') as f:
  for docs_ in tqdm(docs.docs):
    tags = [str(feeds[doc]['manual_tag_list']).replace(';', ' ') for doc in docs_]
    tags = set(' '.join(tags).split())
    print(' '.join(tags), file=f)
    tags2 = [' '.join([x.split()[0] for x in str(feeds[doc]['machine_tag_list']).split(';')]) for doc in docs_]
    tags2 = set(' '.join(tags2).split())
    print(' '.join(tags2), file=f)


# In[287]:


with open('../input/key_valid_corpus.txt', 'w') as f:
  for docs_ in tqdm(docs.docs):
    tags = [str(feeds[doc]['manual_keyword_list']).replace(';', ' ') for doc in docs_]
    tags = set(' '.join(tags).split())
    print(' '.join(tags), file=f)
    tags2 = [' '.join([x.split()[0] for x in str(feeds[doc]['machine_keyword_list']).split(';')]) for doc in docs_]
    tags2 = set(' '.join(tags2).split())
    print(' '.join(tags2), file=f)


# In[288]:


docs = docs_test


# In[289]:


with open('../input/author_test_corpus.txt', 'w') as f:
  for docs_ in tqdm(docs.docs):
    authors = [feeds[doc]['authorid'] for doc in docs_]
    print(' '.join(map(str, authors)), file=f)
with open('../input/singer_test_corpus.txt', 'w') as f:
  for docs_ in tqdm(docs.docs):
    singers = [feeds[doc]['bgm_singer_id'] for doc in docs_]
    print(' '.join(map(str, singers)), file=f)
with open('../input/song_test_corpus.txt', 'w') as f:
  for docs_ in tqdm(docs.docs):
    songs = [feeds[doc]['bgm_song_id'] for doc in docs_]
    print(' '.join(map(str, songs)), file=f)


# In[290]:


with open('../input/tag_test_corpus.txt', 'w') as f:
  for docs_ in tqdm(docs.docs):
    tags = [str(feeds[doc]['manual_tag_list']).replace(';', ' ') for doc in docs_]
    tags = set(' '.join(tags).split())
    print(' '.join(tags), file=f)
    tags2 = [' '.join([x.split()[0] for x in str(feeds[doc]['machine_tag_list']).split(';')]) for doc in docs_]
    tags2 = set(' '.join(tags2).split())
    print(' '.join(tags2), file=f)


# In[291]:


with open('../input/key_test_corpus.txt', 'w') as f:
  for docs_ in tqdm(docs.docs):
    tags = [str(feeds[doc]['manual_keyword_list']).replace(';', ' ') for doc in docs_]
    tags = set(' '.join(tags).split())
    print(' '.join(tags), file=f)
    tags2 = [' '.join([x.split()[0] for x in str(feeds[doc]['machine_keyword_list']).split(';')]) for doc in docs_]
    tags2 = set(' '.join(tags2).split())
    print(' '.join(tags2), file=f)


# In[ ]:





# In[ ]:





# In[ ]:




