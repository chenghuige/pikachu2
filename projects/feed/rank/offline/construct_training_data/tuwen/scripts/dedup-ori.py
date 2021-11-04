#!/usr/bin/env python 
# -*- coding: utf-8 -*-
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import json
import traceback
from config import *

from pyspark import SparkConf, SparkContext

conf = SparkConf() \
    .set("spark.ui.showConsoleProgress", "true") \
    .set('spark.hive.mapred.supports.subdirectories', 'true') \
    .set('spark.hadoop.mapreduce.input.fileinputformat.input.dir.recursive', 'true') \
    .set('spark.executor.memory','4g') \
    .set("spark.default.parallelism", '500') \
    .set('spark.dynamicAllocation.enabled', 'true') \
    .set('spark.port.maxRetries', '100') 

sc = SparkContext(conf=conf)

ifile = sys.argv[1]
ofile = sys.argv[2]

# input is utf8 already and must not set use_unicode=False
d = sc.textFile(ifile)

d_ori = d

def parse(line):
  try:
    l = line.split('\t')
    # mid\tdocid as key
    key = '\t'.join(l[1:3])
    click = l[0]
    product = l[3]
    info = l[4]
    # ignore show_time l[5]
    other = '\t'.join(l[6:])
    r = [key, (click, product, info, other)]
    return r
  except Exception:
    return None

d = d.map(parse).filter(lambda x: x is not None)

d = d.groupByKey()

def dedup(l):
  # try:
  key = l[0]
  show_info = None
  click_info = None
  other = None
  clicked = '0'
  product = 'sgsapp'
  for item in l[1]:
    click, product_, info, other_ = item
    click = item[0]
    product_ = item[1]
    info = item[2]
    other_ = item[3]
    if click == '1':
      clicked = '1'
      click_info = info
      if show_info is not None:
        break
    else:
      show_info = info
      other = other_
      product = product_
      if click_info is not None:
        break

  # if click then merge click info to show info
  if click_info:
    if not show_info:
      return ''
    click_info = json.loads(click_info)
    show_info = json.loads(show_info)
    for key_ in click_info:
      show_info[key_] = click_info[key_]
    show_info = json.dumps(show_info, ensure_ascii=True)
  
  line = '\t'.join([clicked, key, product, show_info, other])
  return line

d = d.map(dedup)
d2 = d.filter(lambda x: x != '')

if DEBUG:
  dedupd_count = d2.count()
  click_no_show_count = d.filter(lambda x: x == '').count()
  print('click_no_show_count', click_no_show_count)
  click_count = d2.filter(lambda x: x.startswith('1')).count()

  print('click_count', click_count, 'click_ratio', click_count / dedupd_count)
  print('click no show ratio', click_no_show_count / d.count(), click_count, click_no_show_count / click_count)
  ori_count = d_ori.count()
  print('dedup_ratio(left)', dedupd_count / ori_count, dedupd_count, ori_count)
try:
  d2.saveAsTextFile(ofile, compressionCodecClass=COMPRESS)
except Exception as e:
  pass

