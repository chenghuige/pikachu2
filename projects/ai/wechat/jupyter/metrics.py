#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import glob
import sys,os
sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')
from multiprocessing import Pool, Manager, cpu_count 
import pymp
import qgrid
# import plotly
# import plotly.graph_objs as go
# from plotly.subplots import make_subplots
# import plotly.io as pio
# pio.renderers.default = "jupyterlab"
import gezi
from gezi import tqdm, line
tqdm.pandas()
from IPython.display import display

import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# In[2]:


get_ipython().run_cell_magic('html', '', '<style>\n.output_wrapper, .output {\n    height:auto !important;\n    max-height:10000px;  /* your desired max-height here */\n}\n.output_scroll {\n    box-shadow:none !important;\n    webkit-box-shadow:none !important;\n}\n</style>')


# In[3]:


def gen_df(root):
  dfs= Manager().list()
  pattern = f'{root}/*/metrics.csv'
  files = glob.glob(pattern)
  if not files:
    return None
  files = sorted(files, key=lambda x: os.path.getmtime(x))
  ps = min(len(files), cpu_count())
  with pymp.Parallel(ps) as p:
#     for i in tqdm(p.range(len(files)),desc='gen_df'):
    for i in p.range(len(files)):
      file = files[i]
      if not gezi.non_empty(file):
        continue
      df = pd.read_csv(file)
      df['model'] = os.path.basename(os.path.dirname(file))
      df['mtime'] = os.path.getmtime(file)
      df['ctime'] = os.path.getctime(file)
      df['step'] = [x + 1 for x in range(len(df))]
      dfs.append(df)
  df = pd.concat(list(dfs))
  return df


# In[4]:


def gen_history(root):
  dfs= Manager().list()
  pattern = f'{root}/*/history.csv'
  files = glob.glob(pattern)
  if not files:
    return None
  files = sorted(files, key=lambda x: os.path.getmtime(x))
  ps = min(len(files), cpu_count())
  with pymp.Parallel(ps) as p:
#     for i in tqdm(p.range(len(files)),desc='gen_history', leave=False):
    for i in p.range(len(files)):
      file = files[i]
      if not gezi.non_empty(file):
        continue
      df = pd.read_csv(file)
      df['model'] = os.path.basename(os.path.dirname(file))
      df['mtime'] = os.path.getmtime(file)
      df['ctime'] = os.path.getctime(file)
      df['step'] = [x + 1 for x in range(len(df))]
      dfs.append(df)
  df = pd.concat(list(dfs))
  return df


# In[5]:


v = 0
mark = 'offline'
# mark = 'online'


# In[6]:


root = f'../working/{mark}/{v}'


# In[7]:


history = gen_history(root)


# In[8]:


# history.groupby('model')['epoch'].max().reset_index()


# In[9]:


# history


# In[10]:


# history.head()


# In[11]:


keys = ['train_loss', 'val_loss', 'lr']
line(history, keys, x='step', color='model', smoothing=0.8)


# In[12]:


def show_loss():
  return history.groupby(['step', 'model'])['val_loss']     .aggregate(np.mean).reset_index()     .pivot('step', 'model', 'val_loss')


# In[13]:


# show_loss()


# In[14]:


def show(key, action='score'):
  metric = f'{key}/{action}'
  res = df.groupby(['step', 'model'])[metric]     .aggregate(np.mean).reset_index()     .pivot('step', 'model', metric)
  figs = line(df, metric, x='step', color='model', return_figs=True)
  for fig in figs:
    display(fig)
  return res


# In[15]:


df = gen_df(root)


# In[16]:


df[['model', 'step', 'all/score']].sort_values(['all/score'], ascending=False).head(10)


# In[17]:


# df.groupby(['step', 'model'])['all/score'].max().reset_index().head()


# In[18]:


line(df, 'all/score', x='step', color='model')


# In[19]:


show('all')


# In[20]:


show('all', 'loss')


# In[21]:


show('all', 'read_comment')


# In[22]:


show('test_a')


# In[23]:


show('test_b')


# In[24]:


show('hotdoc')


# In[25]:


show('colddoc')


# In[26]:


show('hot')


# In[27]:


show('cold')


# In[28]:


show('first')


# In[29]:


abc


# In[ ]:


ACTIONS = [
  'read_comment',
  'like',
  'click_avatar',
  'forward',
  'favorite',
  'comment',
  'follow'
]


# In[ ]:


for action in ['score'] + ACTIONS:
  key = f'all/{action}'
  display(df.groupby('model')[key].max().reset_index().sort_values(key, ascending=False))


# In[ ]:


for prefix in ['first', 'cold', 'hot', 'colddoc', 'hotdoc', 'user0', 'user1']:
  key = f'{prefix}/score'
  if key in df.columns:
    display(df.groupby('model')[key].max().reset_index().sort_values(key, ascending=False))


# In[ ]:


models = [(x, df[df.model==x]['ctime'][0]) for x in set(df.model)]
models.sort(key=lambda x: -x[-1])
models_ = [x[0] for x in models[:30]]
models_


# In[ ]:


metrics = ['all/score']
df_ = df[df.model.isin(models_)]
df_[['model', 'step'] + metrics]


# In[ ]:


line(df_, metrics, x='step', color='model')


# In[ ]:





# In[ ]:





# In[ ]:




