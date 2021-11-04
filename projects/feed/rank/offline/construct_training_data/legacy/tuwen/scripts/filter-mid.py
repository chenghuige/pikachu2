#!/usr/bin/env python 
# -*- coding: utf-8 -*-
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

from pyspark import SparkConf, SparkContext

conf = SparkConf() \
        .set("spark.ui.showConsoleProgress", "true") \
	.set('spark.hive.mapred.supports.subdirectories', 'true') \
        .set('spark.hadoop.mapreduce.input.fileinputformat.input.dir.recursive', 'true') \
        .set('spark.executor.memory','1g') \
        .set("spark.default.parallelism", '500') \
        .set('spark.dynamicAllocation.enabled', 'true') \
        .set('spark.port.maxRetries', '100') 

sc = SparkContext(conf=conf)

#ifile = sys.argv[1]
#mid = sys.argv[2]

#ifile = 'hdfs://GodSonNameNode2/user/traffic_dm/fujinbing/real_show_feature_new/20191120/2019112*'
ifile = 'hdfs://GodSonNameNode2/user/traffic_dm/fujinbing/real_show_feature_new/20191122/*'
#ifile = 'chg/rank/video_hour_sgsapp_v1/gen_feature/20191122*'
mid = 'f113864728033179897'
docid  = '19b19a042w0por'

d = sc.textFile(ifile)

def filter(line):
  l = line.strip().split()
  if len(l) < 3:
    return False
  mid_, docid_, click = line.split()[:3]
  #click, mid_, docid_, = line.split()[:3]
  #return mid_ == mid
  return docid_ == docid and click != '0'

d = d.filter(filter)

def parse(line):
  #return '\t'.join(line.split()[:3])
  return '\t'.join(line.split())

d = d.map(parse)
res = d.collect()

with open('/tmp/all22.%s.txt' % docid, 'w') as out:
  for item in res:
    print(item, file=out)
