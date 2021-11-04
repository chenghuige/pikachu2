#!/usr/bin/env python 
# -*- coding: utf-8 -*-

## this is filter of one hour original data(after dedup) 
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import json
import random
from config import *

from pyspark import SparkConf, SparkContext

conf = SparkConf() \
    .set("spark.ui.showConsoleProgress", "true") \
    .set('spark.hive.mapred.supports.subdirectories', 'true') \
    .set('spark.hadoop.mapreduce.input.fileinputformat.input.dir.recursive', 'true') \
    .set('spark.executor.memory','3g') \
    .set("spark.default.parallelism", '500') \
    .set('spark.dynamicAllocation.enabled', 'true') \
    .set('spark.port.maxRetries', '100') 

sc = SparkContext(conf=conf)

ifile = sys.argv[1]
# ifile += '/part-00000'
ofile = sys.argv[2]
ifile2 = sys.argv[4] #read 24h mid-docid history
ofile2 = sys.argv[3] #store cur-1h mid-docid

d = sc.textFile(ifile)
d_ori = d

FIELDS = 3

def instace_filter(line):
  l = line.split('\t')
  
  info = l[4]  
  try:
    behavior = json.loads(info)
  except Exception as e:
    return False
  
  if MARK != ALL_MARK:
    user_interest_cnt = behavior.get("interest_cnt")
    ty = str(user_interest_cnt.get("ty", "1"))

    if ty != MARK:
      return False
  
  return True

# Notice we filter tuwen or video at first since our user may only interst in tuwen or video
d = d.filter(instace_filter)

def mids_parse(line):
  l = line.split('\t')
  key = l[1]
  show = 1
  click = int(l[0])
  non_click = 1 - click
  val = [show, click, non_click]
  r = [key, val]
  return r

def add(x, y):
  r = [0] * FIELDS
  for i in range(FIELDS):
    r[i] = x[i] + y[i]
  return r

# all_mids = d.map(mids_parse).reduceByKey(lambda x, y: x + y) 
all_mids = d.map(mids_parse)


if READ_MID_DOCID_HISTORY and ifile2 != 'None':
    print('read_mid_docid_history!')
    d3 = sc.textFile(ifile2)
    d3 = d3.map(mids_parse)
    all_mids = all_mids.union(d3)
all_mids = all_mids.reduceByKey(add)
max_show = MAX_SHOW 
# bad_mids = all_mids.filter(lambda l: (l[1][0] >  max_show or l[1][0] < 3) or (l[1][1] == 0) or (l[1][2] == 0))
# bad_mids_set = set(bad_mids.map(lambda l: l[0]).collect())

# TODO might try Broadcast Hash Join ? as all mids is a small table
all_mids_map = dict(all_mids.collect())

if DEBUG:
  num_bad_mids = bad_mids.count()
  num_all_mids = all_mids.count()
  print('bad mids ratio:', num_bad_mids / num_all_mids, num_bad_mids, num_all_mids)
  bad_mids_maxshow = all_mids.filter(lambda l: l[1][0] >  max_show).count()
  bad_mids_minshow = all_mids.filter(lambda l: l[1][0] < 3).count()
  bad_mids_no_click = all_mids.filter(lambda l: l[1][1] == 0).count()
  bad_mids_only_click = all_mids.filter(lambda l: l[1][2] == 0).count()
  print(bad_mids_maxshow, bad_mids_minshow, bad_mids_no_click, bad_mids_only_click)
  
  # so filter in one hour data has too many no click ..
  # bad mids ratio: 0.748541107308 294128 392935                                    
  # 46 6303 293799 54


def mid_filter(line):
  line_tuple = line.split('\t')
  
  if len(line_tuple) < 3:
    return False
  
  (click, mid, docid, product, info) = line_tuple[0:5]
  
  if mid not in all_mids_map:
    return False 
  
  show, click, non_click = all_mids_map[mid]
  
  if show > max_show or show < 3:
    return False
  
  # ignore all click user instances
  if not non_click:
    return False
   
  try:
    behavior = json.loads(info)
    is_ceil = int(behavior.get("is_ceil", 0))
    if is_ceil == 1:
      return True
  except Exception:
    pass
  
  # we hold neg and pos instances for user who clicked in 1 hour
  if click:
    return True
    
  # wether to random append neg instances, by default will not as FILTER_RATIO is 0
  if random.random() < FILTER_RATIO:
    return True

  return False

d = d.filter(mid_filter)

if DEBUG:
  print('after filter', d.count())

# def prepare_join(line):
#   l = line.split('\t')
#   key = l[1]
#   val = line.rstrip()
#   return [key, val]

# def deal_join(l):
#   key = l[0]
#   left = l[1][0]
#   right = l[1][1]
#   user_show = right[0]
#   left = left.split('\t')
#   info = json.loads(left[4], encoding='utf8')
#   info['user_show'] = user_show 
#   left[4] = json.dumps(info, ensure_ascii=True)
  
#   return '\t'.join(left)

# d = d.map(prepare_join).join(all_mids).map(deal_join)


if DEBUG:
  print('---all mids map')

def add_info(line):
  l = line.split('\t')
  key = l[1]
  user_info = all_mids_map.get(key)
  if user_info:
    info = json.loads(l[4], encoding='utf8')
    info['user_show'] = user_info[0] 
    l[4] = json.dumps(info, ensure_ascii=True)
    return '\t'.join(l)
  else:
    return line
  
d = d.map(add_info)
  
def add_mid(line):
    l = line.split('\t')
    return l[0]+'\t'+l[1]+'\t'+l[2]

if STORE_MID_DOCID:
    print('store_mid_docid!')
    d4 = d_ori.map(add_mid)
    try:
        d4.saveAsTextFile(ofile2, compressionCodecClass=COMPRESS)
    except Exception as e:
        pass

if DEBUG:
  num_ori_instances = d_ori.count()
  num_filtered_instances = d.count()
  print('filter ratio:', num_filtered_instances / num_ori_instances, 'final', num_filtered_instances, 'ori', num_ori_instances)
  
try:
  d.saveAsTextFile(ofile, compressionCodecClass=COMPRESS)
except Exception as e:
  pass

