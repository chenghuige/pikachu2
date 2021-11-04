#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from collections import OrderedDict
import glob
import sys 
sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')
import gezi
from gezi import tqdm


# In[2]:


d1 = pd.read_csv('../working/offline/v12/neg.finish/valid.csv')


# In[5]:


d1


# In[12]:


d2 = pd.read_csv('../working/offline/v13/base/valid.csv')


# In[13]:


# d2


# In[8]:


gezi.metrics.group_scores(d1.read_comment, d2.read_comment, d1.userid)


# In[14]:


gezi.metrics.group_scores(d1.forward, d2.forward, d1.userid)


# In[9]:


xs = glob.glob('../working/offline/v20/repro*/valid.csv')
print(xs)
d3 = pd.read_csv('../working/offline/v20/repro.')


# In[10]:


d2 = pd.read_csv('../working/offline/v20/repro.2021-06-24_11:05:13/valid.csv')
gezi.metrics.group_scores(d1.read_comment, d2.read_comment, d1.userid)


# In[11]:


gezi.metrics.group_scores(d1.forward, d2.forward, d1.userid)


# In[15]:


d2 = pd.read_csv('../input/user_action.csv')


# In[16]:


d2 = d2.sort_values(['date_'], ascending=False)


# In[17]:


d2[(d2.userid==211090) & (d2.feedid==64543)]


# In[18]:


d2[(d2.userid==43365) & (d2.feedid==17563)]


# In[19]:


d2[(d2.userid==40885) & (d2.feedid==51549)]


# In[20]:


x = gezi.read_pickle('../input/history.pkl')


# In[21]:


d = pd.read_csv('../input/user_action2.csv')


# In[22]:


d[d.is_first == 1]


# In[23]:


d[d.is_first == 0]


# In[ ]:




