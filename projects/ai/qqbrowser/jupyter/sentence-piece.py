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
import sentencepiece as spm
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


# In[2]:


d = pd.read_csv('../input/info/infos.csv')


# In[3]:


with open('../input/corpus.txt', 'w') as f:
  for row in tqdm(d.itertuples(), total=len(d)):
    print(row.title, file=f)
    print(row.asr_text, file=f)


# In[4]:


spm.SentencePieceTrainer.train(input='../input/corpus.txt', model_prefix='../input/sp10w', vocab_size=100000, user_defined_symbols=[f'SEP{i}' for i in range(200)])


# In[5]:


sp = spm.SentencePieceProcessor(model_file='../input/sp10w.model')
sp.encode('This is a test')


# In[6]:


sp.decode([22130,101,102])


# In[7]:


for line in open('../input/corpus.txt'):
  line = line.strip()
  print(line)
  print(sp.encode(line))
  print(sp.encode(line, out_type=str))
  break


# In[8]:


vocab = gezi.Vocab('../input/sp10w.vocab', 0)


# In[9]:


sp.encode('我只不过')


# In[10]:


vocab.id('我')


# In[11]:


vocab.id('光子')


# In[12]:


vocab.key(101)


# In[13]:


vocab_names = [
                'tag',
                'word',
              ]
vocabs = {}
for vocab_name in vocab_names:
  if vocab_name != 'word':
    vocab_file =  f'../input/{vocab_name}_vocab.txt'
    vocab = gezi.Vocab(vocab_file)
  else:
    vocab_file = '../input/sp10w.vocab'
    vocab = gezi.Vocab(vocab_file, 0)
  vocabs[vocab_name] = vocab


# In[14]:


d.head()


# In[15]:


title_words = []
asr_words = []
for row in tqdm(d.itertuples(), total=len(d)):
  title_words.append(len(sp.encode(row.title)))
  asr_words.append(len(sp.encode(str(row.asr_text))))


# In[16]:


d['title_words'] = title_words
d['asr_words'] = asr_words


# In[17]:


d.title_words.describe([.5,.9,.99,.999])


# In[18]:


d.asr_words.describe([.5,.9,.99,.999])


# In[19]:


def gen_w2v(window=32, min_count=1, emb_dim=256):
  sentences = []
  for row in tqdm(d.itertuples(), total=len(d)):
    row = row._asdict()
    l = ['[CLS]', *sp.encode(str(row['title']), out_type=str), '[SEP]', *sp.encode(str(row['asr_text']), out_type=str), '[SEP]']
    sentences.append(l)
  ic(len(sentences))
  name = 'word'
  monitor = gezi.MonitorCallback(name) 
  w2v = Word2Vec(sentences, vector_size=emb_dim, window=window, min_count=min_count, sg=1, workers=mp.cpu_count(), epochs=10, callbacks=[monitor])
  ofile = f'../input/w2v/sp/{emb_dim}/{name}.pkl'
  gezi.try_mkdir(os.path.dirname(ofile))
  gezi.save_pickle(w2v, ofile)
  for name in vocabs:
    vocab = vocabs[name]
    # emb = np.zeros([vocab.size(), emb_dim])
    emb = np.random.uniform(-0.05, 0.05,(vocab.size(), emb_dim))
    for i in range(vocab.size()):
      word = vocab.key(i) 
      if word in w2v.wv:
        emb[i] = w2v.wv[word]
    ofile = f'../input/w2v/sp/{emb_dim}/{name}.npy'
    np.save(ofile, emb)
  
  return w2v


# In[20]:


gen_w2v(emb_dim=256)


# In[21]:


gen_w2v(emb_dim=512)


# In[1]:


#gen_w2v(emb_dim=300)


# In[ ]:


gen_w2v(emb_dim=400)


# In[ ]:


#gen_w2v(emb_dim=600)


# In[22]:


#gen_w2v(emb_dim=768)


# In[ ]:




