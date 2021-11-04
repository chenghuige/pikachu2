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


MAX_TAGS = 6
MAX_KEYS = 7
DESC_LEN = 64
DESC_CHAR_LEN = 128
WORD_LEN = 128
START_ID = 3
UNK_ID = 1
EMPTY_ID = 2
NAN_ID = -1


# In[4]:


d.description.apply(lambda x: len(str(x).split())).describe([.90,.95,.99,.999])


# In[5]:


d = d.fillna(NAN_ID)


# In[6]:


d.head()


# In[29]:


# generate feed_info.npy

MIN_COUNT = None
# MIN_COUNT = 5
    
single_keys = ['author', 'song', 'singer']
multi_keys =  ['manual_tags', 'machine_tags', 'machine_tag_probs', 'machine_tags2', 'manual_keys', 'machine_keys', 'desc', 'desc_char', 'ocr', 'asr']
info_keys = single_keys + multi_keys
info_lens = [1] * len(single_keys) + [MAX_TAGS, MAX_TAGS, MAX_TAGS, MAX_TAGS, MAX_KEYS, MAX_KEYS, DESC_LEN, DESC_CHAR_LEN, WORD_LEN, WORD_LEN]

vocab_names = [
                'user', 'doc',
                'author', 'singer', 'song',
                'key', 'tag', 'word', 'char'
              ]
vocabs = {}

for vocab_name in vocab_names:
  vocab_file =  f'../input/{vocab_name}_vocab.txt'
  # if not doc then mask as UNK for rare words, tags, keys..
  min_count = None if vocab_name == 'doc' else MIN_COUNT
#   min_count = MIN_COUNT if vocab_name in ['word', 'char'] else None
  vocab = gezi.Vocab(vocab_file, min_count=min_count)
  vocabs[vocab_name] = vocab
    
d = pd.read_csv('../input/feed_info.csv')
d = d.fillna(NAN_ID)
d.feedid = d.feedid.astype(int)
d.authorid = d.authorid.astype(int)
d.bgm_singer_id = d.bgm_singer_id.astype(int)
d.bgm_song_id = d.bgm_song_id.astype(int)

def gen_unk():
  l = [0] * sum(info_lens)
  x = 0
  for len_ in info_lens:
    l[x] = UNK_ID
    x += len_
  return l

def gen_empty():
  l = [0] * sum(info_lens)
  x = 0
  for len_ in info_lens:
    l[x] = EMPTY_ID
    x += len_
  return l

embs = [
    gen_empty()
] * vocabs['doc'].size()
embs[0] = [0] * sum(info_lens)
# embs[1] = gen_unk()
embs[1] = [0] * sum(info_lens)
print(np.asarray(embs).shape)

def set_feature(feature, feed, key, feedid):
  key_ = key.replace('description', 'desc')
  if str(feed[key]) != str(NAN_ID):
    feature[key_] = list(map(int, str(feed[key]).split())) 
  else:
    feature[key_] = []

#   MAX_LEN = 512 
  MAX_LEN = WORD_LEN
  if key_ == 'desc':
    MAX_LEN = DESC_LEN
  elif key_ == 'desc_char':
    MAX_LEN = DESC_CHAR_LEN
    
  vocab_name = 'char' if 'char' in key else 'word'
  vocab = vocabs[vocab_name]
  feature[key_] = [vocab.id(x) for x in feature[key_]]
  feature[key_] = gezi.pad(feature[key_], MAX_LEN, 0, last_tokens=10) 

GOOD_TAG_PROB = 0.2

for row in tqdm(d.itertuples(), total=len(d), desc='feed_info'):
  row = row._asdict()
  author = row['authorid']
  feedid = row['feedid']
  song = row['bgm_song_id']
  singer = row['bgm_singer_id']
  l = [
      vocabs['author'].id(author),
      vocabs['song'].id(song),
      vocabs['singer'].id(singer)
  ]
    
  manual_tags = str(row['manual_tag_list']).split(';')
  manual_tags_ = []
  for tag in manual_tags:
    if tag == 'nan' or tag == str(NAN_ID):
      continue
    else:
      manual_tags_.append(vocabs['tag'].id(int(tag)))
  manual_tags_ = gezi.pad(manual_tags_, MAX_TAGS, 0)
  l += manual_tags_
  
  machine_tags = str(row['machine_tag_list']).split(';')
  machine_tags_ = []
  machine_tag_probs_ = []
  machine_tags2_ = []
  for item in machine_tags:
    if item == 'nan' or item == str(NAN_ID):
      continue
    else:
      x = item.split()
      tag, prob = int(x[0]), float(x[1])
      machine_tags_.append(vocabs['tag'].id(tag))
      machine_tag_probs_.append(int(prob * 1e7))
      if prob > GOOD_TAG_PROB:
        machine_tags2_.append(vocabs['tag'].id(tag))
  machine_tags_ = gezi.pad(machine_tags_, MAX_TAGS, 0)
  l += machine_tags_
  machine_tag_probs_ = gezi.pad(machine_tag_probs_, MAX_TAGS, 0)
  l += machine_tag_probs_
  machine_tags2_ = gezi.pad(machine_tags_, MAX_TAGS, 0)
  l += machine_tags2_
  

  manual_keys = str(row['manual_keyword_list']).split(';')
  manual_keys_ = []
  for key in manual_keys:
    if key == 'nan' or key == str(NAN_ID):
      continue
    else:
      manual_keys_.append(vocabs['key'].id(int(key)))
  manual_keys_ = gezi.pad(manual_keys_, MAX_KEYS, 0)
  l += manual_keys_
    
  machine_keys = str(row['machine_keyword_list']).split(';')
  machine_keys_ = []
  for key in machine_keys:
    if key == 'nan' or key == str(NAN_ID):
      continue
    else:
      machine_keys_.append(vocabs['key'].id(int(key)))
  machine_keys_ = gezi.pad(machine_keys_, MAX_KEYS, 0)
  l += machine_keys_
    
  feature = OrderedDict()
#   for key in ['description', 'description_char', 'ocr', 'ocr_char', 'asr', 'asr_char']:
  for key in ['description', 'description_char', 'ocr', 'asr']:
    set_feature(feature, row, key, feedid)
    
  for key in feature:
    l += feature[key]

  embs[vocabs['doc'].id(int(row['feedid']))] = l

print(np.asarray(embs).shape)

embs = np.asarray(embs)

print('--save to ../input/doc_lookup.npy')
np.save('../input/doc_lookup.npy', embs)


# In[30]:


np.asarray(embs).shape


# In[31]:


embs[vocabs['doc'].id(77432)]


# In[32]:


lens = [info_lens[0]] * len(info_lens)
for i in range(len(info_lens)):
  if i > 0:
    lens[i] = info_lens[i] + lens[i - 1]
embs_ = np.split(embs, lens, axis=-1)


# In[27]:


sum(info_lens)


# In[34]:



for key, len_, emb in zip(info_keys, info_lens, embs_):
  print(key, len_, emb.shape)
  np.save(f'../input/{key}_lookup.npy', emb)


# In[ ]:




