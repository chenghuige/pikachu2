#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   dump-day-vids-uids.py
#        \author   chenghuige  
#          \date   2020-06-12 17:17:08.522506
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import os, sys
import glob
import time
from datetime import timedelta, datetime
import pandas as pd
from plotly.graph_objs import Scatter,Layout
import plotly
import plotly.offline as py
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
#setting offilne
plotly.offline.init_notebook_mode(connected=True)
import pyarrow.parquet as pq
from gezi import tqdm
import gezi
import pickle

import pickle
d_cs = []
for i in tqdm(range(30)):
  dir_ = f'../input/train/part_{i+1}'
  d = gezi.read_parquet(f'{dir_}/context.parquet')
  pickle.dump(set(d.did.values), open(f'{dir_}/dids_1.pkl', 'wb'))
  pickle.dump(set(d.vid.values), open(f'{dir_}/vids_1.pkl', 'wb'))
  d_cs += [d]
  dt = pd.concat(d_cs)
  pickle.dump(set(dt.did.values), open(f'{dir_}/dids.pkl', 'wb'))
  pickle.dump(set(dt.vid.values), open(f'{dir_}/vids.pkl', 'wb'))
d_c = pd.concat(d_cs)

dce = gezi.read_parquet('../input/eval/context.parquet')
dc30 = gezi.read_parquet(f'../input/train/part_30/context.parquet')
pickle.dump(set(dc30.did.values), open(f'../input/train/part_30/dids_1.pkl', 'wb'))
pickle.dump(set(dc30.vid.values), open(f'../input/train/part_30/vids_1.pkl', 'wb'))
pickle.dump(set(dce.did.values), open(f'../input/eval/dids_1.pkl', 'wb'))
pickle.dump(set(dce.vid.values), open(f'../input/eval/vids_1.pkl', 'wb'))
dall = pd.concat([dce, d_c])
pickle.dump(set(dall.did.values), open(f'../input/eval/dids.pkl', 'wb'))
pickle.dump(set(dall.vid.values), open(f'../input/eval/vids.pkl', 'wb'))

