import streamlit as st

import os
import glob
import traceback
from datetime import datetime
import pandas as pd
from plotly.graph_objs import Scatter,Layout
import plotly
import plotly.offline as py
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
plotly.offline.init_notebook_mode(connected=True)
import gezi
from gezi import tqdm, line
import pymp
from multiprocessing import Pool, Manager, cpu_count 
from projects.feed.rank.monitor.taurus import Taurus
import qgrid


st.title('tarus-hourly-sgsapp')

last_days = 15
start_time = None
end_time = None
# start_time = 20191210
# end_time = 20191220
# abids = [8,15,16]
# abids = [16, 8]
# abids = [8]
# abids = [8, 15]
# abids = [16, 11]
abids = [15]
product = 'sgsapp'
# product = 'newmse'

diff_spans = {
    8: [20200313, 20200319],
    12: [20190130, 20200205],
    16: [20190130, 20200205],
    11: [20190130, 20200205],
    15: [20200325, 20200331],
    16: [20200325, 20200331],
}
diff_spans = None

last_days = 1
start_time = None
end_time = None
# start_time = 20191210
# end_time = 20191220
# abids = [8, 15, 16]
# abids = [16, 8]
# abids = [8]
worker_hourly = Taurus()
worker_hourly.init(abids, last_days, 'hourly', product=product,
                 diff_spans=diff_spans, start_time=start_time, end_time=end_time)
worker_hourly.run()

# for name 1(tuwen,video,small_video,total) 2(quality,total) 3(recommend,relative,total)
def show_hourly(name, use_diff=True, quality=False, field=None):
  worker_hourly.show(name, quality, field, use_diff=use_diff, relative_diff=True)

show_hourly('total_total_total')

