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

from projects.common.monitor.taurus import Taurus
from projects.common.monitor.show import show

#last_days = 20
last_days = 2
abids = [8,15,16]
abids = [16, 8]
mark = 'hourly'
#mark = 'daily'

mark2 = '小时' if mark == 'hourly' else '天'

start_time = None
end_time = None
# start_time = 20191210
# end_time = 20191220

worker = Taurus()
worker.init(abids, last_days, mark, use_natural_diff=True, start_time=start_time, end_time=end_time)
dfs, diffs = worker.run()

def show_(name):
  return show(dfs, name, worker.stats, abids, mark=mark, diffs=diffs)

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
  for name in worker.names:
    rows += gen_graphs(show_(name))
    # rows += [html.Div(dcc.Graph(figure=x)) for x in show_(name)]
  return rows

def gen_hours():
  print(datetime.now())
  global dfs, diffs
  dfs, diffs = worker.run()
  return [html.H1(f'Taurus{mark2}级 {datetime.now()}')] + gen_rows()

hours_data = gen_hours()
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

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title=f'Taurus{mark2}级'
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True

app.layout = gen_layout

# @app.callback(Output('hours', 'children'),
#               [Input('interval-component', 'n_intervals')])
# def update_graph_live(n):
#   print(f'update {n}')
#   return gen_hours()



if __name__ =='__main__':
  app.run_server(debug=True, host='10.141.202.84', port=8051)  
  # app.run_server(debug=False, host='10.141.202.84')                  


