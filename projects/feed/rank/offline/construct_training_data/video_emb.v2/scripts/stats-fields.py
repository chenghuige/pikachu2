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
        .set('spark.executor.memory','3g') \
        .set("spark.default.parallelism", '500') \
        .set('spark.dynamicAllocation.enabled', 'true') \
        .set('spark.port.maxRetries', '100') 

sc = SparkContext(conf=conf)
ifile = ','.join([os.path.join(x, 'field_ori') for x in sys.argv[1].split(',')])
odir = sys.argv[2]
ofile = '%s/%s' % (odir, 'field_ori')
ofile2 = '%s/%s' % (odir, 'field')

d = sc.textFile(ifile)

def parse(line):
  fname, cnt = line.split('\t')
  return (fname, int(cnt))

d = d.map(parse).reduceByKey(lambda x, y: x + y).sortBy(lambda x: -x[1])
  
d.map(lambda l: '\t'.join([l[0], str(l[1])])).repartition(1).saveAsTextFile(ofile)

d = d.sortByKey(numPartitions=1)
d = d.map(lambda l: l[0])
d.repartition(1).saveAsTextFile(ofile2)
