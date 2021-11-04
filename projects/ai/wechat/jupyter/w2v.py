#!/usr/bin/env python
# coding: utf-8

# In[8]:


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


# In[2]:


ACTIONS = ["read_comment", "like", "click_avatar",  "forward", "comment", "follow", "favorite"]
ACTIONS2 = ACTIONS + ['finish', 'stay']
HIS_ACTIONS = ACTIONS2 + [
    'pos', 'neg', 'unfinish', 'unstay'
]


# In[3]:


# FAST = True
# if FAST:
  


# In[4]:


days = sys.argv[1].split(',')
emb_dim = 128
window = int(sys.argv[2])
# min_count = int(sys.argv[3])
min_count = 1
#if len(sys.argv) > 3:
#  emb_dim = int(sys.argv[3])
sg = 1


# In[5]:


from gensim.models.callbacks import CallbackAny2Vec

class MonitorCallback(CallbackAny2Vec):
  def __init__(self, name):
    self.name = name
    self.epoch = 1
    self.timer = gezi.Timer()
    
  def on_epoch_end(self, model):
    # TODO 为什么打印train loss一直是0
    print('name:', self.name, 'epoch:', self.epoch, "model loss:", model.get_latest_training_loss(), f'elapsed minutes: {self.timer.elapsed_minutes():.2f}')  # print loss
    self.epoch += 1


# In[6]:


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


# In[7]:


def gen_w2v(name, day, window=32, min_count=1, sg=1, emb_dim=128):
  sentences = []
  if day:
    file = f'../input/{day}/{name}_corpus.txt'
  else:
    file = f'../input/{name}_corpus.txt'
  for line in open(file):
    l = line.rstrip().split()
    sentences.append(l)
  monitor = MonitorCallback(name) 
  w2v = Word2Vec(sentences, vector_size=emb_dim, window=window, min_count=min_count, sg=sg, workers=mp.cpu_count(), epochs=10, callbacks=[monitor])
  vocab = vocabs[name]
  emb = np.zeros([vocab.size(), emb_dim])
  #emb = np.random.uniform(-0.05, 0.05,(vocab.size(), emb_dim))
  count = 0
  for i in range(vocab.size()):
    word = vocab.key(i) 
    if word in w2v.wv:
      emb[i] = w2v.wv[word]
      count += 1
  #unk_emb = np.sum(emb, 0) / count
  #for i in range(vocab.size()):
  #  word = vocab.key(i)
  #  if word not in w2v.wv:
  #    emb[i] = unk_emb
  mark = 'w2v' if sg else 'cbow'
  if day:
    ofile = f'../input/{day}/{name}_{mark}_window{window}_emb.npy'
  else:
    ofile = f'../input/{name}_{mark}_window{window}_emb.npy'
  np.save(ofile, emb)
  # print(emb.shape)


# In[8]:


names = [
  'doc', 
  'user',
  'author', 
  'singer', 
  'song',
  ]

# if len(sys.argv) > 4:
#   sg = int(sys.argv[4])

if len(sys.argv) > 3:
  names = sys.argv[3].split(',')

for day in tqdm(days):
  t = tqdm(names, desc=f'{day}')
  for name in t:
    t.set_postfix({'name': name, 'day': day, 'window': window, 'min_count': min_count, 'sg': sg, 'emb_dim': emb_dim})
    gen_w2v(name, day, window=window, min_count=min_count, sg=sg, emb_dim=emb_dim)
    # gen_w2v(name, day, window=32, min_count=1) # 0.6828
    # gen_w2v(name, day, window=32, min_count=5) # 0.6825
    # gen_w2v(name, day, window=64, min_count=1) # 0.6829
    # gen_w2v(name, day, window=128, min_count=1)  # 0.6852
    # gen_w2v(name, day, window=256, min_count=1)


# In[16]:

if not  len(sys.argv) > 4:
  names = [
  'word', 
  'tag',
  # 'key',
  # 'char'
  ]
  for day in tqdm(days):
    t = tqdm(names, desc=f'{day}')
    for name in t:
      t.set_postfix({'name': name})
      gen_w2v(name, 0, window=window, min_count=min_count, sg=sg, emb_dim=emb_dim)


# In[12]:


# d = pd.read_csv('../input/feed_info.csv')
# NAN_ID = -1
# d = d.fillna(NAN_ID)


# In[26]:


# desc_documents = []
# count = 0
# for i, desc in tqdm(enumerate(d.description.values), total=len(d)):
#   if str(desc) == str(NAN_ID):
#     continue
#   desc_documents.append(TaggedDocument(desc.split(), [count]))
#   count += 1
# monitor = MonitorCallback('desc') 
# model = Doc2Vec(desc_documents, vector_size=emb_dim, window=8, min_count=1, workers=mp.cpu_count(), callbacks=[monitor])
# model.save('../input/desc_doc2vec.model')


# In[27]:


# ocr_documents = []
# count = 0
# for i, ocr in tqdm(enumerate(d.ocr.values), total=len(d)):
#   if str(ocr) == str(NAN_ID):
#     continue
#   desc_documents.append(TaggedDocument(ocr.split(), [count]))
#   count += 1
# monitor = MonitorCallback('ocr') 
# model = Doc2Vec(desc_documents, vector_size=emb_dim, window=8, min_count=1, workers=mp.cpu_count(), callbacks=[monitor])
# model.save('../input/ocr_doc2vec.model')


# In[28]:


# asr_documents = []
# count = 0
# for i, asr in tqdm(enumerate(d.asr.values), total=len(d)):
#   if str(asr) == str(NAN_ID):
#     continue
#   desc_documents.append(TaggedDocument(asr.split(), [count]))
#   count += 1
# monitor = MonitorCallback('asr') 
# model = Doc2Vec(desc_documents, vector_size=emb_dim, window=8, min_count=1, workers=mp.cpu_count(), callbacks=[monitor])
# model.save('../input/asr_doc2vec.model')


# In[9]:


# vocab = vocabs['doc']


# In[16]:


# model = Doc2Vec.load('../input/desc_doc2vec.model')
# emb = np.zeros([vocab.size(), emb_dim])
# for i, desc in tqdm(enumerate(d.description.values), total=len(d)):
#   if str(desc) == str(NAN_ID):
#     continue
#   emb[vocab.id(int(d.feedid.values[i]))] = model.infer_vector(desc.split())
# np.save('../input/desc_vec_emb.npy', emb)


# In[17]:


# model = Doc2Vec.load('../input/ocr_doc2vec.model')
# emb = np.zeros([vocab.size(), emb_dim])
# for i, ocr in tqdm(enumerate(d.ocr.values), total=len(d)):
#   if str(ocr) == str(NAN_ID):
#     continue
#   emb[i] = model.infer_vector(ocr.split())
# np.save('../input/ocr_vec_emb.npy', emb)


# In[18]:


# model = Doc2Vec.load('../input/asr_doc2vec.model')
# emb = np.zeros([vocab.size(), emb_dim])
# for i, asr in tqdm(enumerate(d.asr.values), total=len(d)):
#   if str(asr) == str(NAN_ID):
#     continue
#   emb[i] = model.infer_vector(asr.split())
# np.save('../input/asr_vec_emb.npy', emb)


# In[ ]:




