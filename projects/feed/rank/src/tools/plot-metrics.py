#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   plot-metrics.py
#        \author   chenghuige  
#          \date   2019-12-14 09:13:44.320442
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import glob
import pandas as pd
from plotly.graph_objs import Scatter,Layout
import plotly
import plotly.offline as py
import numpy as np
import plotly.graph_objs as go

#setting offilne
plotly.offline.init_notebook_mode(connected=True)

root = sys.argv[1]

def gen_datas(root):
  datas = []
  for dir in glob.glob('%s/*' % root):
    if os.path.isdir(dir):
        name = os.path.basename(dir)
        metric_file = os.path.join(dir, 'metric_hours.txt')
        if os.path.exists(metric_file):
            df = pd.read_csv(metric_file, sep='\t')
            df.sort_values(by=['hour'])
            
            data = go.Scatter(
                x=df.hour,
                y=df.group_auc,
                mode='lines+markers',
                line_shape='spline',
                line_smoothing=1.3,
                name=name
            )
            
            datas.append(data)
  return datas 

datas = gen_datas(root)
layout = go.Layout(xaxis=dict(type='category'))
fig = go.Figure(data=datas, layout=layout)
plotly.io.write_html(fig, file='%s/metrics.html' % root, auto_play=False)
