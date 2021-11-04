#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict
import glob
import sys, os
sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')
import gezi
from gezi import tqdm


# In[14]:


USER_ACTION = '../input/user_action.csv'
ROOT_PATH = '../input'
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar",  "forward", "comment", "follow", "favorite"]
END_DAY = 15


# In[19]:


def statis_feature(start_day=1, before_day=7, agg='sum'):
    """
    统计用户/feed 过去n天各类行为的次数
    :param start_day: Int. 起始日期
    :param before_day: Int. 时间范围（天数）
    :param agg: String. 统计方法
    """
    history_data = pd.read_csv(USER_ACTION)[["userid", "date_", "feedid"] + FEA_COLUMN_LIST]
#     feature_dir = os.path.join(ROOT_PATH, "feature")
    feature_dir = ROOT_PATH
    for dim in ["userid", "feedid"]:
        print(dim)
        user_data = history_data[[dim, "date_"] + FEA_COLUMN_LIST]
        res_arr = []
        for start in tqdm(range(start_day, END_DAY-before_day+1)):
            temp = user_data[((user_data["date_"]) >= start) & (user_data["date_"] < (start + before_day))]
            temp = temp.drop(columns=['date_'])
            temp = temp.groupby([dim]).agg([agg]).reset_index()
            temp.columns = list(map(''.join, temp.columns.values))
            temp["date_"] = start + before_day
            res_arr.append(temp)
        dim_feature = pd.concat(res_arr)
        feature_path = os.path.join(feature_dir, dim+"_feature.csv")
        print('Save to: %s'%feature_path)
        dim_feature.to_csv(feature_path, index=False)


# In[20]:


statis_feature()


# In[21]:


d = pd.read_csv('../input/feedid_feature.csv')


# In[22]:


d.head()


# In[23]:


d[d.feedid == 1]


# In[24]:


set(d.date_)


# In[25]:


his = pd.read_csv('../input/user_action.csv')


# In[26]:


his[his.feedid==1]


# In[ ]:




