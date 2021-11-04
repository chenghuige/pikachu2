#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict
import multiprocessing as mp
import glob
import sys, os
sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')
import gezi
from gezi import tqdm
tqdm.pandas()
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import jieba


# In[2]:


vocab_names = [
                'tag',
                'word',
              ]
vocabs = {}
for vocab_name in vocab_names:
  vocab_file =  f'../input/{vocab_name}_vocab.txt'
  if vocab_name != 'word':
    vocab = gezi.Vocab(vocab_file)
  else:
    vocab = gezi.Vocab(vocab_file, 200)
  vocabs[vocab_name] = vocab


# In[3]:


vocabs['tag']


# In[4]:


d = pd.read_csv('../input/info/infos.csv')


# In[5]:


d.head(10)


# In[6]:


def gen_w2v(name, window=32, min_count=1, emb_dim=256):
  sentences = []
  for row in tqdm(d.itertuples(), total=len(d)):
    row = row._asdict()
    if name == 'word':
      l = ['[CLS]', *jieba.cut(str(row['title'])), '[SEP]', *jieba.cut(str(row['asr_text'])), '[SEP]']
    else:
      l = str(row[name]).split(',')
    sentences.append(l)
  ic(len(sentences))
  name = name.replace('_id', '')
  monitor = gezi.MonitorCallback(name) 
  w2v = Word2Vec(sentences, vector_size=emb_dim, window=window, min_count=min_count, sg=1, workers=mp.cpu_count(), epochs=10, callbacks=[monitor])
  ofile = f'../input/w2v/jieba/{emb_dim}/{name}.pkl'
  gezi.try_mkdir(os.path.dirname(ofile))
  vocab = vocabs[name]
  # emb = np.zeros([vocab.size(), emb_dim])
  emb = np.random.uniform(-0.05, 0.05,(vocab.size(), emb_dim))
  for i in range(vocab.size()):
    word = vocab.key(i) 
    if word in w2v.wv:
      emb[i] = w2v.wv[word]
  ofile = f'../input/w2v/jieba/{emb_dim}/{name}.npy'
  np.save(ofile, emb)
  
  return w2v


# In[8]:


# gen_w2v('tag_id', emb_dim=256)


# In[ ]:


# gen_w2v('tag_id', emb_dim=512)


# In[ ]:


gen_w2v('word', emb_dim=256)


# In[ ]:

gen_w2v('word', emb_dim=512)


# In[7]:


#  gen_w2v('word', emb_dim=768)


# In[8]:


#gen_w2v('word', emb_dim=300)


# In[ ]:


gen_w2v('word', emb_dim=400)


# In[ ]:


#gen_w2v('word', emb_dim=600)

