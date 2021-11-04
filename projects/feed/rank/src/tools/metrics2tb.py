#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   metrics2tb.py
#        \author   chenghuige  
#          \date   2019-12-15 21:10:53.711717
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import tensorflow as tf
from absl import app, flags
FLAGS = flags.FLAGS

flags.DEFINE_string('base_dir', '/search/odin/publicData/CloudS/rank/infos/', '')
flags.DEFINE_integer('base_abid', 16, '')
flags.DEFINE_string('root', './', '')
flags.DEFINE_string('mark', None, 'video or tuwen')
flags.DEFINE_string('product', 'sgsapp', '')

import gezi
from gezi import SummaryWriter

import os
import glob
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import multiprocessing
import numpy as np
import traceback

dfs = []
tb_dir = None 
hours2step = {}
start_hour = 1e32
end_hour = -1

def gen_dfs(root):
	global start_hour, end_hour
	base_records = []
	for dir in tqdm(glob.glob(f'{root}/*')):
		if os.path.isdir(dir):
			try:
				df = pd.read_csv(f'{dir}/metric_hours.csv')
				df = df.groupby(df.hour, as_index=False).last().sort_values(by=['hour'])
				df = df.fillna(0.)
				df.name = os.path.basename(dir)
				print(df.name)
				dfs.append(df)
				if df.hour.max() > end_hour:
					end_hour = df.hour.max()
				if df.hour.min() < start_hour:
					start_hour = df.hour.min()
				df_record = pd.read_csv(f'{dir}/base_metric_hours.csv')
				df_record = df_record.fillna(0.)
				base_records.append(df_record)
			except Exception:
				# print(traceback.format_exc())
				continue
	df_record = pd.concat(base_records)
	df_record = df_record.groupby(df_record.hour, as_index=False).last().sort_values(by=['hour'])
	df_record.name = 'record'
	dfs.append(df_record)
	return dfs
					

def write_summary(x):
	df, name = x
	# TODO in evaluate do not change group/ .. to group_
	def rename(key):
		key = key.replace('Dur_', 'Dur/').replace('Click_', 'Click/').replace('group_', 'AGroup/')
		if not key[0].isupper():
			key = 'All/' + key
		return key

	# for df in tqdm(dfs, ascii=True):
	# TODO why df.name not find when parrallel ?
	# print(df.name)
	print(name)

	writer = SummaryWriter(f'{tb_dir}/{name}', is_tf=False)
	hour_idx = None
	for i in tqdm(range(len(df.columns)), ascii=True):
		if df.columns[i] == 'hour':
			hour_idx = i
	assert hour_idx is not None
	
	bad_keys = set()
	for key in df.columns:
		if key == 'index' or key == 'hour':
			continue
		if df[key].max() == df[key].min():
			bad_keys.add(key)

	for row in df.itertuples():
		hour = row[hour_idx + 1]
		step = hours2step[hour]
		for i in range(len(df.columns)):
			key = df.columns[i]
			if key == 'hour':
				continue
			
			if key == 'online':
				continue

			if key == 'gold_auc':
				continue

			if key in bad_keys:
				continue 

			key = rename(key)

			try:
				val = float(row[i + 1])
			except Exception:
				continue
			
			if val != np.nan:
			  writer.scalar(key, val, step, walltime=datetime.strptime(str(hour), '%Y%m%d%H').timestamp())


def deal_metric_file(metric_file, name, abid):
	if metric_file and os.path.exists(metric_file):
		base = pd.read_csv(metric_file)
		base = base[base.abtest==abid]
		base = base.groupby(base.hour, as_index=False).last() 
		
		base = base[base.hour >= start_hour][base.hour <= end_hour].sort_values(by=['hour'])
		base.name = name
		return base
	return pd.DataFrame()

def main(_):
	if not FLAGS.mark:
		FLAGS.mark = 'video' if 'video' in os.path.realpath(FLAGS.root) else 'tuwen'

	global dfs
	dfs = gen_dfs(FLAGS.root)

	# offline_metric_file = f'{FLAGS.base_dir}/{FLAGS.mark}/{FLAGS.base_abid}/{FLAGS.product}_metrics_offline.csv'
	# onlie_metric_file = offline_metric_file.replace('offline', 'online')
	# base_offline = deal_metric_file(offline_metric_file, 'base_offline', 45600)
	# base_online = deal_metric_file(onlie_metric_file, 'base_online', 456)
	# if len(base_offline):
	# 	dfs.append(base_offline)
	# if len(base_online):
	# 	dfs.append(base_online)

	for df in dfs:
		df['AAA/sqrt/auc'] =  (df['auc'] * df['click/time_auc']) ** 0.5
		df['AAA/sqrt/group/auc'] =  (df['group/auc'] * df['group/click/time_auc']) ** 0.5

	hours = set()
	for df in dfs:
		for hour in df.hour:
			hours.add(hour)
	hours = sorted(list(hours))
	global tb_dir
	global hours2step
	hours2step = dict(zip(hours, map(lambda x: x + 1, range(len(hours)))))

	tb_dir = '%s/tb' % FLAGS.root
	if os.path.exists(tb_dir):
	  os.system('rm -rf %s' % tb_dir)
	os.system('mkdir -p %s' % tb_dir)

	dfs_ = [(x, x.name) for x in dfs]
	pool = multiprocessing.Pool()
	pool.map(write_summary, dfs_)
	pool.close()
	pool.join()

if __name__ == '__main__':
  app.run(main)  
