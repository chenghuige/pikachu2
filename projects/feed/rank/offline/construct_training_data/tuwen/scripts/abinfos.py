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
        .set("spark.default.parallelism", '500') \
        .set('spark.dynamicAllocation.enabled', 'true') \
        .set('spark.port.maxRetries', '100') 

sc = SparkContext(conf=conf)

baseids = set(sys.argv[1].split(','))
destids = set([sys.argv[2]])
ty = sys.argv[3]

ifile = sys.argv[4]  
# ofile = sys.argv[5]

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

def deal(d_show, d_click, ids):
  d_show = d_show.filter(lambda x: x[1][0] in ids).map(lambda x: x[0]).distinct().map(lambda x: [x, 1])
  d_click = d_click.join(d_show)
  
  show_count = d_show.count()
  click_count = d_click.count()
  total_dur = d_click.map(lambda x: x[1][0]).sum()
  total_user = d_show.map(lambda x: x[0].split('\t')[0]).distinct().count()
  click_user = d_click.map(lambda x: x[0].split('\t')[0]).distinct().count()
  
  result = dict(
                ctr=click_count / show_count,
                read_ratio=click_user / total_user,
                read_dur1=total_dur / total_user,
                read_dur2=total_dur / click_user
                )
  return result


d_show = d.filter(lambda x: x.split('\t', 1)[0] == '0') \
          .map(parse_show) \
          .filter(lambda x: x[1][1] == ty).cache()
          
d_click = d.filter(lambda x: x.split('\t', 1)[0] == '1').map(parse_click).filter(lambda x: x[1] > 0).cache()

base_result = deal(d_show, d_click, baseids)
base_result['info'] = 'online:%s' % sys.argv[1]
dest_result = deal(d_show, d_click, destids)
dest_result['info'] = 'abid:%s' % sys.argv[2]

diff = dict([(key, (dest_result[key] - base_result[key]) / base_result[key]) for key in dest_result if key != 'info'])
diff['info'] = 'diff'
# gezi.pprint_df(pd.DataFrame.from_dict([online_result, result, diff]), importants, print_fn=logging.info)


keys = ['ctr', 'read_ratio', 'read_dur1', 'read_dur2']

results = []
pre = sys.argv[1]
for key in keys:
  results.append('%s_%s:%.5f' % (pre, key, base_result[key]))  
pre = sys.argv[2]
for key in keys:
  results.append('%s_%s:%.5f' % (pre, key, dest_result[key]))  

pre = 'diff'
for key in keys:
  results.append('%s_%s:%.5f' % (pre, key, diff[key]))  
  
print('\n'.join(results))


# print(result, file=open(ofile, 'w'))
