#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   plotly.py
#        \author   chenghuige  
#          \date   2020-01-26 06:54:34.759512
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import glob
from datetime import datetime
import pandas as pd
import numpy as np
import traceback
import math

import sklearn
import seaborn as sns
from io import BytesIO  
import matplotlib.pyplot as plt
from matplotlib import gridspec
import itertools
from skimage import transform

try:
  import plotly
  from plotly.graph_objs import Scatter,Layout
  import plotly.graph_objs as go
  from plotly.subplots import make_subplots
  # import plotly.offline as py
  # plotly.offline.init_notebook_mode(connected=True)
  from plotly.offline import iplot
  import plotly.express as px
  import plotly.tools as tls
  import chart_studio.plotly as py
except Exception:
  pass

import gezi
logging = gezi.logging

# 如果使用 py.iplot colab必须使用下面,直接使用iplot不需要
def enable_plotly_in_cell():
  try:
    import IPython
    from plotly.offline import init_notebook_mode
    display(IPython.core.display.HTML('''<script src="/static/components/requirejs/require.js"></script>'''))
    init_notebook_mode(connected=False)
  except Exception:
    pass

def tobytes(fig=None):
  s = BytesIO()
  if not fig:
    plt.savefig(s, format='png', bbox_inches='tight')
  else:
    fig.savefig(s, format='png', bbox_inches='tight')
  plt.close()
  return s

# def plot_confusion_matrix(cm, class_names=None, info=''):
#   """
#   Returns a matplotlib figure containing the plotted confusion matrix.

#   Args:
#     cm (array, shape = [n, n]): a confusion matrix of integer classes
#     class_names (array, shape = [n]): String names of the integer classes
#   """
#   figure = plt.figure(figsize=(8, 8))
#   # plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
#   plt.imshow(cm, interpolation='nearest', cmap='YlGnBu')
#   # plt.title("Confusion matrix")
#   plt.colorbar()
#   if class_names:
#     tick_marks = np.arange(len(class_names))
#     plt.xticks(tick_marks, class_names, rotation=45)
#     plt.yticks(tick_marks, class_names)

#   # Normalize the confusion matrix.
#   cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=4)

#   # Use white text if squares are dark; otherwise black.
#   threshold = cm.max() / 2.
#   for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#     color = "white" if cm[i, j] > threshold else "black"
#     plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

#   plt.tight_layout()
#   plt.ylabel('True label')
#   plt.xlabel(f'Predicted label\n{info}')
#   return figure

def plot_confusion_matrix(cm, classes=None, info='', title='', img_size=None):
  if img_size:
    plt.figure(figsize=(img_size, img_size))
  # TODO 似乎不能关闭。。。 notebook 还是会绘图哪怕show=False 可能sns.heatmap默认是show
  ax = sns.heatmap(cm, annot=True, cmap='YlGnBu')
  if title:
    ax.set_title(title)
  ax.set(ylabel='True label', xlabel=f'Predicted label\n{info}')
  if classes is not None and len(classes):
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes, rotation=45) # TODO set 90 not work... 
  
  fig = plt.gcf()
  return fig

def _normalize(cm, normalize=None):
  if normalize:
    axis = None
    if normalize == 'true':
      axis = 1
    elif normalize == 'pred':
      axis = 0
    elif normalize == 'all':
      axis = None
    else:
      raise ValueError(axis)

    cm = cm.astype('float') 
    total = cm.sum(axis=axis, keepdims=(axis != None))
    cm /= total
  return cm

def calc_plot_confusion_matrix(y_true, y_pred, classes=None, normalize=None, info='', title='', img2bytes=False):
  cm = sklearn.metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
  cm = _normalize(cm, normalize)
  fig = plot_confusion_matrix(cm, classes, info, title)
  if img2bytes:
    fig = tobytes(fig)
  return fig

def calc_confusion_matrix(y_true, y_pred, classes=None, normalize=None, info='', title='', img2bytes=False):
  cm = sklearn.metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
  cm = _normalize(cm, normalize)  
  fig = plot_confusion_matrix(cm, classes, info, title, show=show)
  if img2bytes:
    fig = tobytes(fig)
  return fig

def confusion_matrix(cm, classes=None, normalize=None, info='', title='', img2bytes=False, img_size=None):
  '''
  normalize{‘true’, ‘pred’, ‘all’}, default=None
  Normalizes confusion matrix over the true (rows), predicted (columns) conditions or all the population. If None, confusion matrix will not be normalized.
  '''
  cm = _normalize(cm, normalize)
  fig = plot_confusion_matrix(cm, classes, info, title, img_size=img_size)
  if img2bytes:
    fig = tobytes(fig)
  return fig

def create_label_colormap():
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


def label_to_color_image(label):
    """
    添加颜色到图片，根据数据集标签的颜色映射 label colormap

    参数:
        label: 整数类型的 2D 数组array, 保存了分割的类别标签 label

    返回:
        result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.

    Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')
    
    return colormap[label]

def segmentation(image, seg_map, classes, prob=None, title=None, fontsize=35, color_remap=True, img2bytes=False):
    """
    输入图片和分割 mask 的可视化.
    """
    classes = np.asarray(classes)
    FULL_COLOR_MAP = gezi.get('FULL_COLOR_MAP')
    if FULL_COLOR_MAP is None or color_remap:
      FULL_LABEL_MAP = np.arange(len(classes)).reshape(len(classes), 1)
      FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)
      gezi.set('FULL_COLOR_MAP', FULL_COLOR_MAP)

    plt.figure(figsize=(15, 5))
    if prob is None:
      grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])
    else:
      grid_spec = gridspec.GridSpec(1, 5, width_ratios=[6, 6, 6, 6, 1])

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('image')

    plt.subplot(grid_spec[1])
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('mask')

    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('overlay')

    if prob is not None:
      plt.subplot(grid_spec[0, 3])
      display_heatmap(image, 1 - prob, alpha=0.6)
      plt.axis('off')
      plt.title('heatmap_(1-prob)')  

    unique_labels = np.unique(seg_map)
    ax = plt.subplot(grid_spec[-1])
    plt.imshow(FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), classes[unique_labels], fontsize=fontsize)
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid('off')

    if title:
      plt.suptitle(title, size=fontsize)

    fig = plt.gcf()
    if img2bytes:
      fig = tobytes(fig)
    return fig

def segmentation_eval_logits(image, seg_map, logits, classes, preds=None, **kwargs):
    probs = gezi.softmax(logits, -1)
    return segmentation_eval_probs(image, seg_map, probs, classes, preds=preds, **kwargs)

def segmentation_eval_probs(image, seg_map, probs, classes, preds=None, **kwargs):
  pred = preds if preds is not None else probs.argmax(-1)
  seg_map = seg_map.astype(np.int32)
  prob_label = gezi.lookup_nd(probs, seg_map)
  prob = probs.max(-1)
  return segmentation_eval(image, seg_map, pred, classes, prob, prob_label, **kwargs)

# TODO 事实上在colab所有的show=True都不work
def segmentation_eval(image, seg_map, pred_map, classes,  prob=None, prob_label=None, probs=None, title=None, fontsize=35, color_remap=True, img2bytes=False):
    """
    输入图片和分割 mask 的可视化.
    """
    classes = np.asarray(classes)
    FULL_COLOR_MAP = gezi.get('FULL_COLOR_MAP')
    if FULL_COLOR_MAP is None or color_remap:
      FULL_LABEL_MAP = np.arange(len(classes)).reshape(len(classes), 1)
      FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)
      gezi.set('FULL_COLOR_MAP', FULL_COLOR_MAP)

    plt.figure(figsize=(16, 10))
    if prob is None:
      gs = gridspec.GridSpec(2, 4, width_ratios=[6, 6, 6, 2])
    else:
      gs = gridspec.GridSpec(2, 5, width_ratios=[6, 6, 6, 6, 2])

    if probs is not None:
      prob = probs.max(-1)
      prob_label = gezi.lookup_nd(probs, seg_map)

    plt.subplot(gs[0, 0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('image')

    plt.subplot(gs[0, 1])
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('mask_label')

    plt.subplot(gs[0, 2])
    pred_image = label_to_color_image(pred_map).astype(np.uint8)
    plt.imshow(pred_image)
    plt.axis('off')
    plt.title('mask_pred')

    if prob_label is not None:
      plt.subplot(gs[0, 3])
      display_heatmap(image, prob_label, alpha=0.6)
      plt.axis('off')
      plt.title('heatmap_(label-prob)')     

    plt.subplot(gs[1, 0])
    diff_image = (pred_map - seg_map).astype(np.bool).astype(np.uint8)
    plt.imshow(image)
    plt.imshow(diff_image, alpha=0.7)
    plt.axis('off')
    plt.title('overlay_diff')

    plt.subplot(gs[1, 1])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('overlay_label')

    plt.subplot(gs[1, 2])
    plt.imshow(image)
    plt.imshow(pred_image, alpha=0.7)
    plt.axis('off')
    plt.title('overlay_pred')

    if prob is not None:
      plt.subplot(gs[1, 3])
      display_heatmap(image, 1 - prob, alpha=0.6)
      plt.axis('off')
      plt.title('heatmap_(1-prob)')   

    unique_labels = np.unique([*np.unique(seg_map), *np.unique(pred_map)])
    ax = plt.subplot(gs[:, -1])
    plt.imshow(FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), classes[unique_labels], fontsize=int(fontsize/2))
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid('off')

    if title:
      plt.suptitle(title, size=fontsize)
    
    fig = plt.gcf()
    if img2bytes:
      fig = tobytes(fig)
    return fig

def _display_image(image, title, subplot, red=False, titlesize=40):
    plt.subplot(*subplot)
    plt.axis('off')
    try:
      plt.imshow(image)
    except Exception as e:
      logging.error(e)
      logging.error(image)

    if len(title) > 0:
        plt.title(title, fontsize=int(titlesize) if not red else int(titlesize/1.2), color='red' if red else 'black', fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))
    
    return (subplot[0], subplot[1], subplot[2]+1)
    
def display_images(images, title='', titles=[], spacing=0.05, figsize=13., num_images=None, augment=None, show_original=True, img2bytes=False):  
    if not isinstance(images, (list, tuple)):
      try:
        if os.path.isdir(images):
          images = glob.glob(f'{images}/*')
          num_images = num_images or 9 if not augment or not show_original else 6
          np.random.shuffle(images)
          images = images[:num_images]
        else:
          images = [images]
      except Exception:
        images = [images]

    if isinstance(images[0], str):
      import cv2
      images = [cv2.imread(x, cv2.IMREAD_UNCHANGED) for x in images]
      try:
        images = [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in images]
      except Exception:
        pass

    if augment is not None:
      images2 = [augment(x) for x in images]
      if not show_original:
        images = images2
      else:
        l = []
        for i in range(len(images)):
          l.append(images[i])
          l.append(images2[i])
        images = l
    
    # auto-squaring: this will drop data that does not fit into square or square-ish rectangle
    rows = int(math.sqrt(len(images)))
    cols = len(images)//rows
        
    # size and spacing
    FIGSIZE = figsize
    SPACING = spacing # from 0.1 to 0.05
    subplot=(rows,cols,1)
    if rows < cols:
        plt.figure(figsize=(FIGSIZE, FIGSIZE / cols*rows))
    else:
        plt.figure(figsize=(FIGSIZE / rows * cols, FIGSIZE))
    
    titles_ = [''] * len(images) if not titles else titles
    dynamic_titlesize = ((FIGSIZE * SPACING / max(rows,cols) * 40 + 3) * 1.2) # magic formula tested to work from 1x1 to 10x10 images
    # dynamic_titlesize = ((FIGSIZE * SPACING / max(rows,cols) * 30 + 3) * 1.2) # magic formula tested to work from 1x1 to 10x10 images
    # display
    for i, image in enumerate(images[:rows*cols]):
      subplot = _display_image(image, titles_[i], subplot, titlesize=dynamic_titlesize)
    
    #layout
    # #plt.tight_layout()
    # if not titles:
    #     plt.subplots_adjust(wspace=0, hspace=0)
    # else:
    plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
    
    plt.suptitle(title, size=dynamic_titlesize)
    
    fig = plt.gcf()
    
    if img2bytes:
      fig = tobytes(fig)
    return fig

# https://github.com/durandtibo/heatmap.git
def display_heatmap(image, heat_map, alpha=0.6, display=False, normalize=False, save=None, cmap='viridis', axis='on', verbose=False):
  
    height = image.shape[0]
    width = image.shape[1]

    # resize heat map
    heat_map_resized = transform.resize(heat_map, (height, width))

    if normalize:
      # normalize heat map
      max_value = np.max(heat_map_resized)
      min_value = np.min(heat_map_resized)
      normalized_heat_map = (heat_map_resized - min_value) / (max_value - min_value)
    else:
      normalized_heat_map = heat_map_resized

    # display
    plt.imshow(image)
    # plt.imshow(255 * normalized_heat_map, alpha=alpha, cmap=cmap)
    plt.imshow(normalized_heat_map, alpha=alpha, cmap=cmap)
    plt.colorbar()
    plt.axis(axis)

    if display:
        plt.show()

    if save is not None:
        if verbose:
            print('save image: ' + save)
        plt.savefig(save, bbox_inches='tight', pad_inches=0)

def line(dfs, y=[], x='step', color=None, names=None, smoothing=0., line_smoothing=0., include=None, exclude=None, 
         base=True, limit=0, focus=False, return_figs=False, focus_name=None, max_steps=0, upscale=False, show_hovertext=True,
         show_time_delta=True, show_time=True, time_pattern=None, mode='lines+markers'):  
  if dfs is None:
    return 
  metrics = y
  if isinstance(metrics, str):
    metrics = metrics.split(',')
  
  if names and isinstance(names, str):
    names = names.split(',')

  if not isinstance(dfs, (list, tuple)):
    if color is not None:
      if color not in dfs.columns:
        color = 'name'
      models = set(dfs[color].values)
      l = []
      for model in models:
        df = dfs[dfs[color]==model]
        df.name = model
        l += [df]
      dfs = l
    else:
      dfs = [dfs]

  if focus:
    for i in range(len(dfs)):
      dfs[i][x] = dfs[i][x].astype(int)
      
    df = dfs[0]
    if focus_name:
      for i in range(len(dfs)):
        try:
          if dfs[i].name == focus_name:
            df = dfs[i]
            break
        except Exception:
          try:
            if dfs[i][color].values[0] == focus_name:
              df = dfs[i]
              break
          except Exception:
            pass
      
    start = df[x].min()
    end = df[x].max()

    for i in range(len(dfs)):
      name = None
      if hasattr(dfs[i], 'name'):
        name = dfs[i].name
      dfs[i] = dfs[i][dfs[i][x]>=start][dfs[i][x]<=end]
      if name:
        dfs[i].name = name

  if not metrics:
    metrics = dfs[0].columns
  metrics = [x for x in metrics if not x in ['mtime', 'ctime', 'hour', 'step']]

  figs = []
  for metric in metrics:
    try:
      datas = []
      for df in dfs:
        if names and not df.name in names:
          continue
        if not base and df.name == 'baseline':
          continue
        if include and include not in df.name and not df.name == 'baseline':
          continue
        if exclude:
          ok = True
          for x in exclude.split(','):
            if x in metric:
              ok = False
              break
          if not ok:
            continue
        if not len(df):
          if 'name' in df.columns:
            print(df.name, ' is empty', file=sys.stderr)
          continue
        if metric not in df.columns:
          print(metric, 'not found', file=sys.stderr)
          continue
        if df[metric].values.max() == df[metric].values.min() and len(df) > 1:
          continue
        # df[x] = df[x].astype(str)
        df = df.sort_values([x])
        
        d = df.copy()
        d = d[d[metric].notnull()]
        
        if not len(d):
          continue
        
        xs = d[x].values if not limit else d[x].values[-limit:]
        if time_pattern is None:
          if len(str(xs[0])) == len('20191229'):
            xs = [datetime.strptime(str(x), '%Y%m%d') for x in xs]
          elif len(str(xs[0])) == len('2019122912'):
            xs = [datetime.strptime(str(x), '%Y%m%d%H') for x in xs]
        else:
          xs = [datetime.strptime(str(x), time_pattern) for x in xs]

        ys = d[metric].values 

        if max_steps:
          if len(xs) > max_steps:
            # 1,2 1,2,3,4,5,6 -> choose 3,6
            interval = int(len(xs) / max_steps)
            indexes = np.asarray([i for i in range(len(xs)) if xs[i] % interval == 0 or i == len(xs) - 1])
            xs = [x + 1 for x in range(len(indexes))]
            ys = ys[indexes]
          elif len(xs) < max_steps and upscale:
            interval = int(max_steps / len(xs))
            xs = [x * interval for x in xs]

        ys = ys[-limit:]

        # mode = 'lines+markers'
        hovertext = None
        if 'ntime' in df.columns and show_hovertext and (show_time_delta or show_time):
          try:
            ntimes = [pd.to_datetime(x) for x in df['ntime'].values]
            if show_time:
              hovertext = [[ntimes[i].strftime("%a %b %d,%H:%M")] for i in range(len(df))]
            if show_time_delta:
              start_time = pd.to_datetime(df['ctime'].values[0], unit='s') if 'ctime' in df.columns else ntimes[0]
              hovertext = [[*hovertext[i], gezi.format_time_delta(ntimes[i] - start_time)] for i in range(len(df))]
            hovertext = [' '.join(x) for x in hovertext]
          except Exception:
            hovertext = None
        try:
          data = go.Scatter(
            x=xs,
            y=gezi.smooth(ys, smoothing),
            mode=mode,
            line_shape='spline',
            line_smoothing=line_smoothing,
            marker=dict(size=4),
            name=df[color].values[0] if color else 'None',
            hoverlabel=dict(namelength = -1),
            hovertext=hovertext,
          )
        except Exception as e:
          print(traceback.format_exc())
          print(e)
          continue

        datas.append(data)

      if datas:
        if x in ['day', 'hour', 'time']:
          layout = go.Layout(xaxis=dict(type='date'), title=metric, hovermode='x')
        else:
          layout = go.Layout(title=metric, hovermode='x')
        fig = go.Figure(data=datas, layout=layout)
        if not return_figs:
          iplot(fig)
        figs += [fig]
    except Exception as e:
      print(traceback.format_exc())
      print(e)
      pass

  if return_figs:
    return figs

def train_val_loss(history, title=None, smoothing=0.8, line_smoothing=0.):
  if title is None:
    title = 'Model loss'
    if isinstance(history, dict):
      if 'model' in history:
        model = history['model']
        title += f': {model}' 
    else:
      if 'model' in history.columns:
        model = history['model'].values[0]
        title += f': {model}' 
  mode = 'lines+markers'
  loss = go.Scatter(
    x=history['step'].values,
    y=history['loss'].values,
    mode=mode,
    line_shape='spline',
    line_smoothing=line_smoothing,
    marker=dict(size=4),
    name='loss',
    hoverlabel=dict(namelength = -1),
  )
  val_loss = go.Scatter(
    x=history['step'].values,
    y=gezi.smooth(history['val_loss'].values, smoothing),
    mode=mode,
    line_shape='spline',
    line_smoothing=0.5,
    marker=dict(size=4),
    name='val_loss',
    hoverlabel=dict(namelength = -1),
  )
  layout = go.Layout(title=title, hovermode='x')
  fig = go.Figure(data=[loss, val_loss], layout=layout)
  iplot(fig)

history_loss = train_val_loss

def model_loss(history=None, title=None):
  if history is None:
    history = gezi.get('history')
  plt.plot(history['loss'])
  plt.plot(history['val_loss'])

  if title is None:
    title = 'Model loss'
    if isinstance(history, dict):
      if 'model' in history:
        model = history['model']
        title += f': {model}' 
    else:
      if 'model' in history.columns:
        model = history['model'].values[0]
        title += f': {model}'  

  plt.title(title)
  plt.ylabel('loss')
  plt.xlabel('eval_step')
  plt.legend(['Train', 'Valid'], loc='upper right')
  fig = plt.gcf()
  return fig

# def model_metric(metric='loss', history=None, title=None):
#   if history is None:
#     history = gezi.get('history')
#   plt.plot(history[metric])
#   plt.plot(history[f'val_{metric}'])

#   if title is None:
#     title = f'Model {metric}'
#     if isinstance(history, dict):
#       if 'model' in history:
#         model = history['model']
#         title += f': {model}' 
#     else:
#       if 'model' in history.columns:
#         model = history['model'].values[0]
#         title += f': {model}'  

#   plt.title(title)
#   plt.ylabel(metric)
#   plt.xlabel('eval_step')
#   plt.legend(['Train', 'Valid'], loc='upper right')
#   plt.show()

def models_loss(history=None, title=None):
  if history is None:
    history = gezi.get('history')
  models = set(history['model'])
  for model in models:
    history_ = history[history.model==model]
    model_loss(history_, title)

def model_metric(history, metric, models=[]):
  if not models:
    models = set(history['model'])
  for model in models:
    plt.plot(history[history.model==model][metric])
  plt.title(metric)
  plt.ylabel(metric)
  plt.xlabel('eval_step')
  plt.legend(models, loc='bottom right')
  plt.show()  

def model_metrics(history, metrics, models=[]):
  if not models:
    models = set(history['model'])
  if not isinstance(metrics, (list, tuple)):
    metrics = [metrics]
  for metric in metrics:
    model_metric(history, metric, models)
