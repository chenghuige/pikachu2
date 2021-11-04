#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict, ChainMap, Counter
import glob
import sys 
import functools
sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')
import dask.dataframe as dd
from multiprocessing import Pool, Manager, cpu_count
from joblib import Parallel, delayed
import pymp
import gezi
from gezi import tqdm
tqdm.pandas()


# In[2]:


ACTIONS = [
  'read_comment',
  'like',	
  'click_avatar',	
  'forward',
  'favorite',
  'comment',	
  'follow'
]
DAYS = 15


# In[3]:


d = pd.read_csv('../input/user_action2.csv')
d.feedid = d.feedid.astype(int)
d.userid = d.userid.astype(int)
d.date_ = d.date_.astype(int)


# In[4]:


d = d.sort_values(['date_'], ascending=True)


# In[5]:


d.head()


# In[6]:


d.read_comment.mean()


# In[6]:


dates = d.groupby(['feedid'])['date_'].progress_apply(list).reset_index(name='dates')


# In[7]:


dates['dates'] = dates.dates.apply(lambda x:dict(Counter(x)))


# In[8]:


dates


# In[9]:


dates['start_day'] = dates.dates.apply(lambda x: min(x.keys()))


# In[10]:


dates


# In[40]:


doc_dynamic_feature = {}
for feedid in d.feedid.values:
  doc_dynamic_feature[int(feedid)] = {}


# In[41]:


days = DAYS
for row in tqdm(dates.itertuples(), total=len(dates), desc='shows'):
  row = row._asdict()
  dates_ = row['dates']
  shows = [0] * (days + 1)
  for i in range(days):
    i += 1
    if i in dates_:
      shows[i] = dates_[i]
  doc_dynamic_feature[int(row['feedid'])]['shows'] = shows


# In[42]:


doc_dynamic_feature[d.feedid.values[0]]


# In[27]:


def gen_doc_dynamic(d, feedids=None):
  if feedids is not None:
    d = d[d.feedid.isin(set(feedids))]
  dg = d.groupby(['feedid', 'date_'])
  actions = ACTIONS + ['actions', 'finish_rate', 'stay_rate']
  doc_dynamic_feature = {}
  for feedid in d.feedid.values:
    doc_dynamic_feature[int(feedid)] = {}
  t = tqdm(actions)
  for action in t:
    t.set_postfix({'action': action})
    da = dg[action].progress_apply(sum).reset_index(name=f'{action}_count')
    days = DAYS
    for row in tqdm(da.itertuples(), total=len(da), desc=f'{action}_count'):
      row = row._asdict()
      date = row['date_']
      feedid = int(row['feedid'])
      ddf = doc_dynamic_feature[int(row['feedid'])]
      if action not in ddf:
        ddf[action] = [0] * (days + 1)

      ddf[action][date] = row[f'{action}_count']
  return doc_dynamic_feature


# In[17]:


# gen_doc_dynamic(d)


# In[28]:


import pymp
nw = cpu_count()
feedids_list = np.array_split(list(set(d.feedid)), nw)
res = Manager().dict()
with pymp.Parallel(nw) as p:
  for i in p.range(nw):
    res[i] = gen_doc_dynamic(d, feedids_list[i])


# In[30]:


doc_dynamic_feature2 = dict(ChainMap(*res.values()))


# In[43]:


for feedid in doc_dynamic_feature:
  doc_dynamic_feature[feedid].update(doc_dynamic_feature2[feedid])


# In[44]:


doc_dynamic_feature[d.feedid.values[0]]


# In[52]:


dates[dates.feedid==36523]


# In[49]:


doc_dynamic_feature[36523]


# In[45]:


gezi.save_pickle(doc_dynamic_feature, '../input/doc_dynamic_feature.pkl')


# In[46]:


d.finish_rate.mean()


# In[47]:


d.stay_rate.mean()


# In[ ]:


dates.to_csv('../input/doc_static_feature.csv', index=False)


# In[ ]:


dates2 = d.groupby(['userid'])['date_'].progress_apply(list).reset_index(name='dates')


# In[ ]:


dates2['dates'] = dates2.dates.apply(lambda x:dict(Counter(x)))


# In[ ]:


dates2


# In[ ]:


d.groupby(['date_'])['userid'].count()


# In[ ]:


d.groupby(['date_'])['feedid'].count()


# In[ ]:


import numpy as np

def wilson_ctr(clks, imps, z=1.96):
    
    origin_ctr = clks * 1.0 / imps
    
    if origin_ctr > 0.9:
        return 0.0
    
    n = imps
    
    first_part_numerator = origin_ctr + z**2 / (2*n)
    second_part_numerator_2 = np.sqrt(origin_ctr * (1-origin_ctr) / n + z**2 / (4*(n**2)))
    common_denominator = 1 + z**2 / n
    second_part_numerator = z * second_part_numerator_2
    

    new_ctr = (first_part_numerator-second_part_numerator)/common_denominator
    
    return new_ctr

test_case = [(5, 10), (50, 100), (500, 1000), (5000, 10000)]
for item in test_case:
    print(wilson_ctr(*item))


# In[ ]:


import numpy
import random
import scipy.special as special
 
class BayesianSmoothing(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
    
    def sample(self, alpha, beta, num, imp_upperbound):
        # 先验分布参数
        clicks = []
        exposes = []
        for clk_rt in numpy.random.beta(alpha, beta, num):
            imp = imp_upperbound
            clk = int(imp * clk_rt)
            exposes.append(imp)
            clicks.append(clk)
        return clicks, exposes
    
    def update(self, imps, clks, iter_num=1000, epsilon=1e-5):
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(imps, clks, self.alpha, self.beta)
            if abs(new_alpha-self.alpha)<epsilon and abs(new_beta-self.beta)<epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta
            
    def __fixed_point_iteration(self, imps, clks, alpha, beta):
        numerator_alpha = 0.0
        numerator_beta = 0.0
        denominator = 0.0
        
        for i in range(len(imps)):
            numerator_alpha += (special.digamma(clks[i]+alpha) - special.digamma(alpha))
            numerator_beta += (special.digamma(imps[i]-clks[i]+beta) - special.digamma(beta))
            denominator += (special.digamma(imps[i]+alpha+beta) - special.digamma(alpha+beta))
        return alpha*(numerator_alpha/denominator), beta*(numerator_beta/denominator)
    
def main():
    bs = BayesianSmoothing(1, 1)
#     clk, exp = bs.sample(500, 500, 10, 1000)
    clk = [5, 50, 500, 5000]
    exp = [10, 100, 1000, 10000]
    print('原始数据')
    for i, j in zip(clk, exp):
        print(i, j)
        
    bs.update(exp, clk)
    print('bayes光滑先验分布参数：', bs.alpha, bs.beta)
    fixed_ctr = []
    for i in range(len(clk)):
        origin_ctr = clk[i] / exp[i]
        new_ctr = (clk[i] + bs.alpha) / (exp[i]+bs.alpha+bs.beta)
        print('修正前{}, 修正后{}'.format(round(origin_ctr, 3), round(new_ctr, 3)))
    
if __name__ == '__main__':
    main()


# In[ ]:




