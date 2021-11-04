#!/usr/bin/env python 
# -*- coding: utf-8 -*-
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import json
import traceback

from pyspark import SparkConf, SparkContext

conf = SparkConf() \
        .set("spark.ui.showConsoleProgress", "true") \
	      .set('spark.hive.mapred.supports.subdirectories', 'true') \
        .set('spark.hadoop.mapreduce.input.fileinputformat.input.dir.recursive', 'true') \
        .set('spark.executor.memory','6g') \
        .set("spark.default.parallelism", '500') \
        .set('spark.dynamicAllocation.enabled', 'true') \
        .set('spark.port.maxRetries', '100') 

sc = SparkContext(conf=conf)

abtestid = set(sys.argv[1].split(','))

ty = sys.argv[2]
ifile = sys.argv[3]  

d = sc.textFile(ifile)

def parse_show(line):
  reload(sys)
  sys.setdefaultencoding('utf8')
  l = line.split('\t')
  key = '\t'.join(l[1:3])
  info = l[4]
  behavior = json.loads(info.decode('utf8', 'ignore'))
  abtestid = behavior['abtestid']
  user_interest_cnt = behavior.get("interest_cnt")
  ty = str(user_interest_cnt.get("ty", "1"))
  r = [key, (abtestid, ty)]
  return r

  
d_show = d.filter(lambda x: x.split('\t', 1)[0] == '0') \
          .map(parse_show) \
          .filter(lambda x: x is not None and x[1][0] in abtestid and x[1][1] == ty) \
          .map(lambda x: x[0]).distinct().map(lambda  x: [x, 1])
          
show_count = d_show.count()

print('show_count', show_count)


def parse_click(line):
  reload(sys)
  sys.setdefaultencoding('utf8')
  l = line.split('\t')
  key = '\t'.join(l[1:3])
  info = l[4]
  info = json.loads(info.decode('utf8', 'ignore'))
  dur = info.get('dur', '0')
  dur = int(dur)
  if dur > 12 * 60 * 60:
    dur = 60
  r = [key, dur]
  return r

   
d_click = d.filter(lambda x: x.split('\t', 1)[0] == '1').map(parse_click).filter(lambda x: x[1] > 0)

d_click = d_click.join(d_show)

click_count = d_click.count()

print('click_count', click_count)

total_dur = d_click.map(lambda x: x[1][0]).sum()

print('total_dur', total_dur)

total_user = d_show.map(lambda x: x[0].split('\t')[0]).distinct().count()

print('total_user', total_user)

click_user = d_click.map(lambda x: x[0].split('\t')[0]).distinct().count()
print('click_user', click_user)

print(show_count, click_count, \
      'ctr', click_count / show_count, \
      total_user, click_user, 
      'read_ratio', click_user / total_user, \
      'user_dur1', total_dur / total_user, \
      'user_dur2', total_dur / click_user)




