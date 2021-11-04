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

from datetime import datetime, timedelta

from plotly.graph_objs import Scatter,Layout
import plotly
import plotly.offline as py
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import gezi

# setting offilne
# plotly.offline.init_notebook_mode(connected=True)

def gen_figs(df, stats, abids, name='total', quality=False, field=None, data_obj=None, mark='hourly', 
             diffs=None,  diff_ratios=None, use_base=True, smoothing=1., cols=2, compare_days=0, 
             start_time=None, end_time=None, product='sgsapp'):
    # name can be tuwen, video, small_video, total
    # quality or total 
    # field can be recommend, relative, total
    name = name.replace('small_video', 'svideo')
    if name and len(name.split('_')) == 3:
        name, data_obj = name.split('_', 1)
    else:
        if data_obj:
            if len(data_obj.split('_')) == 3:
                name, data_obj = data_obj.split('_', 1)
        else:
            if not field:
                if name == 'total':
                    field = 'total'
                else:
                    field = 'recommend'
            data_obj = f'quality_{field}' if quality else f'total_{field}'

    name = name.replace('all', 'total')
    name = name.replace('svideo', 'small_video')
    data_obj = data_obj.replace('all', 'total')
    df = df[df.name==name]
    df = df[df.data_obj==data_obj]
    assert len(df), f'{name}_{data_obj}, name should in tuwen,video,small_video,total, field should in recommend,relative,total, only sgsapp has quality'

    if diffs is not None:
        diffs = diffs[diffs.name==name]
        diffs = diffs[diffs.data_obj==data_obj]
    elif diff_ratios is not None:
        diff_ratios = diff_ratios[diff_ratios.name==name]
        diff_ratios = diff_ratios[diff_ratios.data_obj==data_obj]

    time_name = 'datetime' if mark == 'hourly' else 'date'

    if start_time:
        if mark == 'daily':
            start_time = int(start_time / 100)
        df = df[df[time_name] >= start_time]
    if end_time:
        if mark == 'daily':
            end_time = int(end_time / 100)
        df = df[df[time_name] <= end_time]
    if isinstance(stats, str):
        stats = stats.split(',')
        
    if isinstance(abids, str):
        abids = [int(x) for x in abids.split(',')]

    assert mark == 'hourly' or mark == 'daily'

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
                diff = diffs[diffs.abtest==abid]
                if mark == 'daily':
                    diff_val = diff[stat].astype(float).values[0]
                    exp_vals -= diff_val
                else:
                    exp_vals = [x - diff[diff.hour==(int(y) % 100)][stat].values[0] \
                         for x, y in zip(df_[stat].values, df_[time_name].values)]
            elif diff_ratios is not None:
                diff_ratio = diff_ratios[diff_ratios.abtest==abid]
                if mark == 'daily':
                    diff_ratio_val = diff_ratio[stat].astype(float).values[0]
                    ratio_val = (exp_vals - base_vals) / base_vals
                    ratio_val -= diff_ratio_val
                    exp_vals = base_vals * (1 + ratio_val)
                else:
                    exp_vals = [base * (1 + (x - base) / base - diff_ratio[diff_ratio.hour==(int(y) % 100)][stat].values[0]) \
                         for x, y, base in zip(df_[stat].values, df_[time_name].values, base_vals)]

            if len(set(exp_vals)) <= 1:
                print(stat, 'with same vals ignore')
                continue
          
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
            if len(set(base_vals)) <= 1:
                continue

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
        

        layout = go.Layout(xaxis=dict(type='date'), title=f'{product}:{name}:{data_obj}:     {stat}', yaxis_title=stat, hovermode='x')
        fig = go.Figure(data=datas, layout=layout)
        # py.iplot(fig)
        figs.append(fig)
    
    return figs
