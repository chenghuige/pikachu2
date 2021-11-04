#!/usr/bin/env python 
# -*- coding: utf-8 -*-
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
from config import *

from pyspark import SparkConf, SparkContext

conf = SparkConf() \
        .set("spark.ui.showConsoleProgress", "true") \
	.set('spark.hive.mapred.supports.subdirectories', 'true') \
        .set('spark.hadoop.mapreduce.input.fileinputformat.input.dir.recursive', 'true') \
        .set('spark.executor.memory','6g') \
        .set("spark.default.parallelism", '500') \
        .set('spark.executor.instances', '500') 
        
sc = SparkContext(conf=conf)

ifile = sys.argv[1]
ofile = sys.argv[2]

d = sc.textFile(ifile)
# d.cache()
# d_ori = d

def parse(line):
  l = line.rstrip().split('\t')
  # mid\tdocid as key
  key = '\t'.join(l[0:2])
  val = '\t'.join(l[2:])
  r = [key, val]
  return r

d = d.map(parse).groupByKey()
# d.cache()

def dedup(l):
  key = l[0]
  val = None
  for item in l[1]:
    click = item.split('\t')[0]
    if click == '1':
      val = item
      break 
  if not val:
    val = item
  return '\t'.join([key, val])

d = d.map(dedup)


d.saveAsTextFile(ofile, compressionCodecClass=COMPRESS)

if DEBUG:
  ori_count = d_ori.count()
  dedupd_count = d.count()
  print('dedup_ratio', dedupd_count / ori_count, dedupd_count, ori_count)
