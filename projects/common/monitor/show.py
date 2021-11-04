#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   show.py
#        \author   chenghuige  
#          \date   2019-12-28 09:05:43.973092
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

from datetime import datetime

from plotly.graph_objs import Scatter,Layout
import plotly
import plotly.offline as py
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

#setting offilne
# plotly.offline.init_notebook_mode(connected=True)

def show(dfs, name, stats, abids, mark='hourly', diffs=None, use_base=True, smoothing=1., cols=2, compare_days=0, start_time=None, end_time=None):
    df = dfs[name]
    if start_time:
        df = df[df[time_name] >= start_time]
    if end_time:
        df = df[df[time_name] <= end_time]
    if isinstance(stats, str):
        stats = stats.split(',')
    stats = [x for x in stats if not (x == 'real_ctr' and 'rel' in name) and \
             not ('refresh' in x and not (name == 'quality' or name == 'all' or name == 'rec'))]
    if isinstance(abids, str):
        abids = [int(x) for x in abids.split(',')]

    assert mark == 'hourly' or mark == 'daily'

    time_name = 'datetime' if mark == 'hourly' else 'date'

    figs = []
    
    for i, stat in enumerate(stats):        
        df4 = df[df.abtest==4]
        df5 = df[df.abtest==5]
        df6 = df[df.abtest==6]   
        base_vals = (df4[stat].values + df5[stat].values + df6[stat].values) / 3.

        datas = []
        for abid in abids:
            df_ = df[df.abtest==abid]    
            exp_vals = df_[stat].astype(float).values
            if diffs is not None:
                diff = diffs[name]
                if mark == 'hourly':
                  diff_ = diff[abid]
                else:
                  diff_ = diff[diff.abtest==abid]
          
                if mark == 'daily':
                    diff_val = diff_[stat].astype(float).values[0]
                    exp_vals -= diff_val
                else:
                    exp_vals = [x + diff_[diff_.hour==(int(y) % 100)][stat].values[0] \
                         for x, y in zip(df_[stat].values, df_[time_name].values)]
          
            diff_vals = exp_vals - base_vals
            ratio_vals = diff_vals / base_vals
            data = go.Scatter(
                x=[datetime.strptime(str(x), '%Y%m%d%H') if len(str(x)) == 10 \
                   else datetime.strptime(str(x), '%Y%m%d') for x in df_[time_name]],
                y=exp_vals,
                mode='lines+markers',
                line_shape='spline',
                hovertext=[str('%.4f' % x) for x in ratio_vals],
                line_smoothing=smoothing,
                showlegend=True,
                name=abid
            )

            datas.append(data)
            
            if compare_days:
                data = go.Scatter(
                    x=[datetime.strptime(str(x), '%Y%m%d%H') + timedelta(compare_days) if len(str(x)) == 10 \
                       else datetime.strptime(str(x), '%Y%m%d') + timedelta(compare_days) for x in df_[time_name]],
                    y=exp_vals,
                    mode='lines+markers',
                    line_shape='spline',
                    hovertext=[str('%.4f' % x) for x in ratio_vals],
                    line_smoothing=smoothing,
                    #                 legendgroup=str(abid),
                    showlegend=True,
                    name=str(abid) + f'({compare_days} before)'
                )

                datas.append(data)
                
        if use_base:
            data = go.Scatter(
                x=[datetime.strptime(str(x), '%Y%m%d%H') if len(str(x)) == 10 \
                   else datetime.strptime(str(x), '%Y%m%d') for x in df5[time_name]],
                y=base_vals,
                mode='lines+markers',
                line_shape='spline',
                line_smoothing=smoothing,
                showlegend=True,
                name='456'
            )   

            datas.append(data)
            
            if compare_days:
                data = go.Scatter(
                    x=[datetime.strptime(str(x), '%Y%m%d%H') + timedelta(compare_days) if len(str(x)) == 10 \
                       else datetime.strptime(str(x), '%Y%m%d') + timedelta(compare_days) for x in df5[time_name]],
                    y=base_vals,
                    mode='lines+markers',
                    line_shape='spline',
                    line_smoothing=smoothing,
                #                 legendgroup='456',
                    showlegend=True,
                    name='456' + f'({compare_days} before)'
                )   

                datas.append(data)
        

        layout = go.Layout(xaxis=dict(type='date'), title=name + ':' + stat, hovermode='x')
        fig = go.Figure(data=datas, layout=layout)
        # py.iplot(fig)
        figs.append(fig)
    
    return figs
