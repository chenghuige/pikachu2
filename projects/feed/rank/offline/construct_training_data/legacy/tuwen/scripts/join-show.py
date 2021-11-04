#!/usr/bin/env python 
# -*- coding: utf-8 -*-
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import json
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

lfile = sys.argv[1]
rfile = sys.argv[2]
# ifile += '/part-00000'
ofile = sys.argv[3]

left = sc.textFile(lfile)
def parse_left(line):
  l = line.rstrip().split('\t')
  key = '\t'.join(l[1:3])
  val = line.rstrip()
  return [key, val]

left = left.map(parse_left)

if DEBUG:
  print('left', left.count())

right = sc.textFile(rfile)
def parse_right(line):
  l = line.rstrip().split('\t')
  r = l[:2]
  key = '\t'.join(r)
  reload(sys)
  sys.setdefaultencoding('utf8')
  # though encoding is gbk assume it to be utf8 as spark not work with gbk
  m = json.loads(l[3].decode('utf8', 'ignore')).get('info', None)
  if m:
    val = (m['ori_lr_score'], m['lr_score'])
  else:
    val = (-1., -1.)

  r = [key, val]
  return r

def select(l):
  key, vals = l
  val = (-1, -1.)
  for val_ in vals:
    if val_[0] != -1.:
      val = val_
      break
  return [key, val]
right = right.map(parse_right).groupByKey().map(select)

if DEBUG:
  print('right', right.count())

def deal_join(l):
  key = l[0]
  left = l[1][0]
  right = l[1][1]
  if right is None:
    right = (-1., -1.)
  ori_lr_score, lr_score = right
  if ori_lr_score > 0:
    left = left.split('\t')
    info = json.loads(left[4])
    info['ori_lr_score'], info['lr_score'] = ori_lr_score, lr_score 
    left[4] = json.dumps(info, ensure_ascii=True)
    return '\t'.join(left)
  else:
    return left
  
d = left.leftOuterJoin(right).map(deal_join)

if DEBUG:
  total = d.count()
  with_lr = d.filter(lambda line: 'ori_lr_score' in line).count()
  d_click = d.filter(lambda line: line.split('\t')[0] == '1')
  total_click = d_click.count()
  with_lr_click = d_click.filter(lambda line: 'ori_lr_score' in line).count()
  print('final', total, 'with_lr', with_lr, 'ratio', with_lr / total, \
        'total_click', total_click, 'with_lr_click', with_lr_click, 'ratio', with_lr_click / total_click)

d.saveAsTextFile(ofile, compressionCodecClass=COMPRESS)

