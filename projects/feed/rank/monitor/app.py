import plotly.graph_objs as go                      
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc                  
import dash_html_components as html                
from dash.dependencies import Input, Output    

import os
import glob
import time
from datetime import timedelta, datetime
import pandas as pd

from projects.feed.rank.monitor.taurus import Taurus
from projects.feed.rank.monitor.conf import diff_spans

from absl import flags
from absl import app as absl_app
FLAGS = flags.FLAGS

flags.DEFINE_string('mark', 'hourly', '')
flags.DEFINE_string('abids', '', '')
flags.DEFINE_integer('last_days', 2, '')
flags.DEFINE_integer('start_time', None, '')
flags.DEFINE_integer('end_time', None, '')
flags.DEFINE_string('ip', '10.141.202.84', '')
flags.DEFINE_string('product', 'sgsapp', '')
flags.DEFINE_string('name', 'total_total_total', '')
flags.DEFINE_integer('compare_days', 0, '')
flags.DEFINE_bool('use_diff', True, '')

worker = None

def show_(name):
  return worker.show(name,use_diff=FLAGS.use_diff, relative_diff=True, 
                     compare_days=FLAGS.compare_days, return_figs=True)

def gen_graphs(figs, cols_per_row=3):
  rows = []
  cols = []
  for fig in figs:
    cols.append(dbc.Col(html.Div(dcc.Graph(figure=fig)), width=4))
    if len(cols) == cols_per_row:
      rows.append(cols)
      cols = []
  if cols:
    rows.append(cols)
  rows = [dbc.Row(x, align='center') for x in rows]
  return rows

def gen_rows():
  rows = []
  names = FLAGS.name.split(',')
  for name in names:
    rows += gen_graphs(show_(name))
    # rows += [html.Div(dcc.Graph(figure=x)) for x in show_(name)]
  return rows

def mark_str():
  return '小时' if FLAGS.mark == 'hourly' else '天'

def gen_hours():
  print(datetime.now())
  worker.run()

  return [html.H1(f'Taurus{mark_str()}级 {datetime.now()}')] + gen_rows()

def gen_layout():
  layout = html.Div([
                      dbc.Container(gen_hours(), fluid=True, id='hours'), 
                      # *gen_hours(),
                      # dcc.Interval(
                      #  id='interval-component',
                      #  interval=10*1000, # in milliseconds
                      #  n_intervals=0) 
                    ])
  return layout

# @app.callback(Output('hours', 'children'),
#               [Input('interval-component', 'n_intervals')])
# def update_graph_live(n):
#   print(f'update {n}')
#   return gen_hours()

def main(_):
  abids = list(map(int, FLAGS.abids.split(','))) if FLAGS.abids else []

  global worker
  worker = Taurus()
  worker.init(abids, FLAGS.last_days, 'hourly', product=FLAGS.product,
              diff_spans=diff_spans, start_time=FLAGS.start_time, end_time=FLAGS.end_time)

  app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
  app.title=f'Taurus{mark_str()}级'
  app.css.config.serve_locally = True
  app.scripts.config.serve_locally = True

  app.layout = gen_layout

  app.run_server(debug=True, host=FLAGS.ip)  

if __name__ == '__main__':
  absl_app.run(main)                 
