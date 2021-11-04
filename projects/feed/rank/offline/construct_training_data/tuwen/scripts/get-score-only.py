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
        .set('spark.executor.instances', '500') \
        .set('spark.port.maxRetries', '100') 

sc = SparkContext(conf=conf)

assert len(sys.argv) > 4

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
  score = str(behavior.get('ori_lr_score', -1.))
  user_interest_cnt = behavior.get("interest_cnt")
  ty = str(user_interest_cnt.get("ty", "1"))
  r = [key, (abtestid, ty, score)]
  return r

  
d_show = d.filter(lambda x: x.split('\t', 1)[0] == '0') \
          .map(parse_show) \
          .filter(lambda x: x is not None and x[1][0] in abtestid and x[1][1] == ty) \
          .map(lambda x: '{}\t{}'.format(x[0], x[1][2])).distinct()
          
d_show.repartition(1).saveAsTextFile(sys.argv[4])

