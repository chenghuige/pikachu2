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


# In[ ]:


d = pd.read_csv('../input/feed_info.csv')


# In[ ]:


NAN_ID = -1
d = d.fillna(NAN_ID)


# In[ ]:


MIN_COUNT = None
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


# In[ ]:


tag_emb = np.load('../input/tag_norm_emb.npy')


# In[6]:


EMB_DIM = 128
embs = [[0] * EMB_DIM] * vocabs['doc'].size()


# In[14]:


for row in tqdm(d.itertuples(), total=len(d), desc='feed_info-manual_tag'):
  row = row._asdict()
  docid = vocabs['doc'].id(int(row['feedid']))
  manual_tags = str(row['manual_tag_list']).split(';')
  manual_tags_embs = []
  for tag in manual_tags:
    if tag == 'nan' or tag == str(NAN_ID):
      continue
    else:
      manual_tags_embs.append(tag_emb[vocabs['tag'].id(int(tag))])
  if manual_tags_embs:
    manual_tags_embs = np.asarray(manual_tags_embs)
    embs[docid] = list(np.mean(manual_tags_embs, 0))


# In[12]:


embs = np.asarray(embs)
print(embs.shape)
np.save('../input/manual_tag_emb.npy', embs)


# In[9]:


embs.shape


# In[10]:


np.load('../input/doc_embs.npy')


# In[23]:


EMB_DIM = 128
embs = [[0] * EMB_DIM] * vocabs['doc'].size()


# In[28]:


for row in tqdm(d.itertuples(), total=len(d), desc='feed_info-machine_tag'):
  row = row._asdict()
  docid = vocabs['doc'].id(int(row['feedid']))
  machine_tags = str(row['machine_tag_list']).split(';')
  machine_tags_embs = []
  for tag in machine_tags:
    if tag == 'nan' or tag == str(NAN_ID):
      continue
    else:
      x = tag.split()
      tag, prob = int(x[0]), float(x[1])
      machine_tags_embs.append(prob * tag_emb[vocabs['tag'].id(int(tag))])
  if machine_tags_embs:
    machine_tags_embs = np.asarray(machine_tags_embs)
    embs[docid] = list(np.mean(machine_tags_embs, axis=0))


# In[29]:


embs = np.asarray(embs)
print(embs.shape)
np.save('../input/machine_tag_emb.npy', embs)


# In[31]:


# embs


# In[15]:


key_emb = np.load('../input/key_norm_emb.npy')


# In[16]:


EMB_DIM = 128
embs = [[0] * EMB_DIM] * vocabs['doc'].size()


# In[18]:


for row in tqdm(d.itertuples(), total=len(d), desc='feed_info-manual_key'):
  row = row._asdict()
  docid = vocabs['doc'].id(int(row['feedid']))
  manual_keys = str(row['manual_keyword_list']).split(';')
  manual_keys_embs = []
  for key in manual_keys:
    if key == 'nan' or key == str(NAN_ID):
      continue
    else:
      manual_keys_embs.append(key_emb[vocabs['key'].id(int(key))])
  if manual_keys_embs:
    manual_keys_embs = np.asarray(manual_keys_embs)
    embs[docid] = list(np.mean(manual_keys_embs, 0))


# In[19]:


embs = np.asarray(embs)
print(embs.shape)
np.save('../input/manual_key_emb.npy', embs)


# In[20]:


EMB_DIM = 128
embs = [[0] * EMB_DIM] * vocabs['doc'].size()


# In[21]:


for row in tqdm(d.itertuples(), total=len(d), desc='feed_info-machine_key'):
  row = row._asdict()
  docid = vocabs['doc'].id(int(row['feedid']))
  machine_keys = str(row['machine_keyword_list']).split(';')
  machine_keys_embs = []
  for key in machine_keys:
    if key == 'nan' or key == str(NAN_ID):
      continue
    else:
      machine_keys_embs.append(key_emb[vocabs['key'].id(int(key))])
  if machine_keys_embs:
    machine_keys_embs = np.asarray(machine_keys_embs)
    embs[docid] = list(np.mean(machine_keys_embs, 0))


# In[22]:


embs = np.asarray(embs)
print(embs.shape)
np.save('../input/machine_key_emb.npy', embs)


# In[38]:


embs = [
    np.load('../input/manual_tag_emb.npy'),
    np.load('../input/manual_key_emb.npy'),
    np.load('../input/machine_tag_emb.npy'),
    np.load('../input/machine_key_emb.npy')
]


# In[39]:


np.save('../input/docs_emb.npy', embs[0])


# In[36]:


embs = np.concatenate(embs, -1)
print(embs.shape)
np.save('../input/docs_emb.npy', embs)


# In[ ]:




