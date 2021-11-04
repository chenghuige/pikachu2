#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os 
sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')
import glob
import pandas as pd
from collections import Counter
import gezi
from gezi import tqdm


# In[ ]:





# In[2]:

# In[5]:


infos = pd.read_csv('../input/infos.csv')
infos['asr_text'] = infos.asr_text.fillna(value='')
infos['category_id'] = infos.category_id.fillna(value=-1)
infos['cat'] = infos.cat.fillna(value=-1)
infos['subcat'] = infos.subcat.fillna(value=-1)
infos['tag_id'] = infos.tag_id.fillna(value='')
infos.category_id = infos.category_id.astype(int)
infos.cat = infos.cat.astype(int)
infos.subcat = infos.subcat.astype(int)
infos = infos.set_index('id')
info = pd.read_csv('../input/info.csv')
info['subject'] = info.subject.fillna(value=-1)
info['subject'] = info.subject.astype(int)
info['tags'] = info.tags.fillna(value='')
info['tag_weights'] = info.tag_weights.fillna(value='')
info = info.set_index('id')


# In[6]:


def sort_tags(tags, tag_weights):
  try:
    tags = tags.split(',')
    tag_weights = [float(x) for x in tag_weights.split(',')]
  except Exception:
    tags = []
    tag_weights = []

  if tags:
    l = zip(tags, tag_weights)
    l = sorted(l, key=lambda x: -x[1])
    tags, tag_weights = zip(*l)
    tags = ','.join(tags)
    tag_weights = ','.join([str(x) for x in tag_weights])
    return tags, tag_weights
  else:
    return '', ''


# In[10]:


tags_list, tag_weights_list = [], []
for row in tqdm(info.itertuples(), total=len(info)):
  x, y = sort_tags(row.tags, row.tag_weights)
  tags_list.append(x)
  tag_weights_list.append(y)
info['tags'], info['tag_weights'] = tags_list, tag_weights_list


# In[ ]:


os.system('mkdir -p ../input/info')


# In[11]:


infos.to_csv('../input/info/infos.csv')


# In[12]:


info.to_csv('../input/info/info.csv')

exit(0)

# In[13]:


infos


# In[14]:


info


# In[11]:


infos.title.map(len).describe(percentiles=[.25,.5,.75,.99]).apply(lambda x: format(x, 'f'))


# In[12]:


infos.asr_text.map(len).describe(percentiles=[.25,.5,.75,.99]).apply(lambda x: format(x, 'f'))


# In[13]:


infos.tag_id.map(lambda x: len(x.split(','))).describe(percentiles=[.25,.5,.75,.99]).apply(lambda x: format(x, 'f'))


# In[14]:


id = 723898876085796218


# In[15]:


infos.loc[id]


# In[16]:


info.loc[id]


# In[17]:


info.index


# In[18]:


# info.loc[1234566]


# In[19]:


tag_names = {}
for row in tqdm(infos.itertuples(), total=len(infos)):
  id = row.Index
  try:
    tag_ids = [int(x) for x in row.tag_id.split(',') if x]
  except Exception:
    # print(row.tag_id)
    continue
  for tag_id in tag_ids:
    if tag_id not in tag_names:
      tag_names[tag_id] = Counter()
    else:
      if id in info.index:
        row_ = info.loc[id]
        tags = row_.tags.split(',')
        for tag in tags:
          tag_names[tag_id][tag] += 1


# In[21]:


gezi.save_pickle(tag_names, '../input/tag_names.pkl')


# In[22]:


with open('../input/tag_names.txt', 'w') as f:  
  for tag in tqdm(tag_names):
    print(tag, tag_names[tag].most_common(3), file=f)


# In[23]:


get_ipython().system('head ../input/tag_names.txt')


# In[20]:


img = f'../input/imgs/{id}.jpg'
title = f'{info.loc[id].title}\n{info.loc[id].tags}'
x = gezi.plot.display_images(img, title=title)
# x = gezi.plot.display_images(img)


# In[2]:


labels = pd.read_csv('../input/pairwise/label.tsv', sep='\t', header=None, names=['query', 'candidate', 'relevance'])
labels['id'] = range(len(labels))
labels['pid'] = labels.apply(lambda x: gezi.hash_str(str(x.query) + str(x.candidate)).upper(), axis=1)
labels['query'] = labels['query'].astype(int)
labels['candidate'] = labels['candidate'].astype(int)
labels['similarity'] = -1.
labels = labels[['id', 'pid', 'query', 'candidate', 'relevance', 'similarity']]


# In[3]:


labels


# In[4]:


labels.to_csv('../input/pairwise/label.csv', index=False)


# In[ ]:


root = '../working/6/model2.ep4.dep8'


# In[ ]:


labels = pd.read_csv(f'{root}/valid.csv')


# In[ ]:


labels


# In[ ]:


MAX_LEN = 15
def wrap_str(title):
  import textwrap
  return '\n'.join(textwrap.wrap(title, MAX_LEN))

def limit_str(text, max_len, last_tokens, sep='|'):
  if len(text) <= max_len:
    return text
  first_tokens = max(max_len - last_tokens - len(sep), 1)
  return text[:first_tokens] + sep + text[-last_tokens:]

def get_text(vid, relevance=-1, similarity=-1):
  tags = wrap_str(info.loc[vid].tags)
  title = wrap_str(infos.loc[vid].title)
  catid = infos.loc[vid].category_id
  tagid = wrap_str(infos.loc[vid].tag_id)
  asr = wrap_str(limit_str(infos.loc[vid].asr_text, 128, 10))
  text = f'id:{vid}\ncat:{catid} relevance:[{relevance}] sim:[{similarity:.2f}]\n-------------------------\n{tags}\n{tagid}\n-------------------------\n{asr}\n-------------------------\n{title}'
  return text


def display_pair(left, right, relevance=0, similarity=0):
  left = int(left)
  right = int(right)
  
  if left in info.index and right in info.index:
    img1 = f'../input/imgs/{left}.jpg'
    img2 = f'../input/imgs/{right}.jpg'
    if os.path.exists(img1) and os.path.exists(img2):
      text1 = get_text(left, relevance, similarity)
      text2 = get_text(right, relevance, similarity)
      _ = gezi.plot.display_images([img1, img2], titles=[text1, text2], spacing=0.05)


# In[ ]:


def display(start, count=10):
  for i in range(start, start + count):
    display_pair(labels.loc[i, 'query'], labels.loc[i, 'candidate'], labels.loc[i, 'relevance'], labels.loc[i, 'similarity'])


# In[ ]:


display(0)


# In[ ]:


def sample_without_replacement(prob_dist, nb_samples):
    """Sample integers in the range [0, N), without replacement, according to the probability
       distribution `prob_dist`, where `N = prob_dist.shape[0]`.
    
    Args:
        prob_dist: 1-D tf.float32 tensor.
    
    Returns:
        selected_indices: 1-D tf.int32 tensor
    """

    nb_candidates = tf.shape(prob_dist)[0]
    logits = tf.math.log(prob_dist)
    z = -tf.math.log(-tf.math.log(tf.random.uniform(shape=[nb_candidates], minval=0, maxval=1)))
    _, selected_indices = tf.math.top_k(logits + z, nb_samples)

    return selected_indices


# In[ ]:


import tensorflow as tf
for i in range(10):
    print('sample {}: {}'.format(i + 1, sample_without_replacement(tf.constant([0.1, 0.2, 0.3, 0.4]), nb_samples=3).numpy()))


# In[ ]:


import heapq as hq
queue = []
hq.heappush(queue, (2, 0, 'a'))
hq.heappush(queue, (3, -1, 'b'))
print(queue)
# [(2, 0, 'a'), (3, -1, 'b')]
hq.heappush(queue, (2, -2, 'c'))
print(queue)
# [(2, -2, 'c'), (3, -1, 'b'), (2, 0, 'a')]


# In[ ]:


print(hq.heappop(queue))
# (2, -2, 'c')
queue
# [(2, 0, 'a'), (3, -1, 'b')]


# In[ ]:


import pandas as pd
d = pd.read_csv('../input/info/infos.csv')


# In[ ]:


6803763629094413742 in d.index


# In[ ]:


6803763629094413742 in d.id


# In[ ]:


d.id


# In[ ]:


5740911444695754158 in d.id.values


# In[ ]:


5740911444695754158 in d.id.values


# In[ ]:


d = d.set_index('id')


# In[ ]:


5740911444695754158 in d.index


# In[ ]:




