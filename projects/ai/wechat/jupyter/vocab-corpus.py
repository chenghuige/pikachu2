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


d = pd.read_csv('../input/feed_info.csv')


# In[3]:


# d.head()


# In[4]:


MAX_TAGS = 14
MAX_KEYS = 18
DESC_LEN = 128
DESC_CHAR_LEN = 256
WORD_LEN = 256
START_ID = 3
UNK_ID = 1
EMPTY_ID = 2
NAN_ID = -1


# In[5]:


d = d.fillna(NAN_ID)


# In[6]:


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


# In[7]:


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


# In[8]:


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


# In[ ]:




