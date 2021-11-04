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
        .set('spark.executor.memory','6g') \
        .set("spark.default.parallelism", '500') \
        .set('spark.executor.instances', '500') \
        .set('spark.port.maxRetries', '100') 

sc = SparkContext(conf=conf)

ifile = sys.argv[1]
ofile = sys.argv[2]

d = sc.textFile(ifile)

def parse(line):
  l = line.split('\t')
  mid = l[MID]
  docid = l[DOCID]
  duration = l[DUR]
  pred = l[ORI_LR_SCORE]
  pred2 = l[LR_SCORE]
  abtestid = l[ABTESTID]
  show_time = l[SHOW_TIME]
  return '\t'.join([mid, docid, abtestid, show_time, duration, pred, pred2])

d.map(parse).saveAsTextFile(ofile)

