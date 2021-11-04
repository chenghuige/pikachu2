#!/usr/bin/env python 
# -*- coding: utf-8 -*-
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import math
from config import *

from pyspark import SparkConf, SparkContext

conf = SparkConf() \
        .set("spark.ui.showConsoleProgress", "true") \
	.set('spark.hive.mapred.supports.subdirectories', 'true') \
        .set('spark.hadoop.mapreduce.input.fileinputformat.input.dir.recursive', 'true') \
        .set('spark.executor.memory','6g') \
        .set("spark.default.parallelism", '500') \
        .set('spark.executor.instances', '500') \
        .set('spark.port.maxRetries', '100') 

sc = SparkContext(conf=conf)

ifile = sys.argv[1]
ofile = sys.argv[2]
interval = int(sys.argv[3])

d = sc.textFile(ifile)
# d.cache()
d_ori = d

FIELDS = 3

def mids_parse(line):
  l = line.rstrip().split('\t')
  key = l[0]
  # val = 1
  click = int(l[CLICK])
  non_click = 1 - click
  val = [1, click, non_click]
  r = [key, val]
  return r

def add(x, y):
  r = [0] * FIELDS
  for i in range(FIELDS):
    r[i] = x[i] + y[i]
  return r

# all_mids = d.map(mids_parse).reduceByKey(lambda x, y: x + y) 
all_mids = d.map(mids_parse).reduceByKey(add) 
# all_mids.cache()

max_show = int(MAX_SHOW * (interval / 24)) if interval > 1 else int(MAX_SHOW * 0.5)
bad_mids = all_mids.filter(lambda l: (l[1][0] > max_show or l[1][0] < MIN_SHOW) or (l[1][1] == 0) or (l[1][2] == 0))
bad_mids_set = set(bad_mids.map(lambda l: l[0]).collect())


def filter(line):
  l = line.rstrip().split('\t')
  l = l[:NUM_PRES]
  # click = l[CLICK]
  # dur = l[DUR]
  mid = l[MID]
  ty = l[TY]

  if MARK != ALL_MARK and ty != MARK:
    return False
  
  if mid in bad_mids_set:
    return False

  return True

d = d.filter(filter)
# def deal_join(l):
#   key = l[0]
#   left = l[1][0]
#   right = l[1][1]
#   user_show = str(right[0])
#   l = [key] + left.split('\t')
#   l1 = l[:NUM_PRES]
#   l2 = l[NUM_PRES:]
#   return '\t'.join(l1 + [user_show] + l2)

# d = d.filter(filter).map(lambda x: x.rstrip().split('\t', 1)).join(all_mids).map(deal_join)
# d.cache()

d.saveAsTextFile(ofile, compressionCodecClass=COMPRESS)

if DEBUG:
  num_bad_mids = bad_mids.count()
  num_all_mids = all_mids.count()

  num_ori_instances = d_ori.count()
  num_filtered_instances = d.count()

  print('bad mids ratio:', num_bad_mids / num_all_mids, num_bad_mids, num_all_mids)
  print('filter ratio:', num_filtered_instances / num_ori_instances, num_filtered_instances, num_ori_instances)

