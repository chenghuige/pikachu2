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
        .set('spark.port.maxRetries', '100') \
        .set('spark.yarn.queue', 'feedflow_online')

sc = SparkContext(conf=conf)

ifile = 'jzh/online1_output/online1_day/online1_2019%s' % sys.argv[1]
abtestids = set(sys.argv[2].split(','))
cold_reasons = set('931,984,925,926'.split(','))

d = sc.textFile(ifile)

def parse(line):
  if(line.find("req")!=0 and line.find("resp")!=0):
    return None

  #try:
  if line.find("req") == 0:
    if len(line.split("\t"))<45:
      return None
    (req, mid, tm, action, topic, mark, title, keywords, userinfo, imsi, url, account, channel, art_source, OS,account_openid, abtestid, sub_topic, image_type, read_duration, position, app_ver, aduser_flag, location,pagetime, rec_reason, adid, vulgar, sub_list, ip, action_source, recall_word, video_type, channel_id, docid, product, source_type,distribution,model,lda_topic,read_completion_rate,xid,ac,a,imei) = line.replace("\n","").split("\t")[0:45]
    if(product!="sgsapp"):
      return None
    if action != '8':
      return None
    if art_source=='15':
      return None
    if channel_id!="1":
      return None
    if video_type != "1":
      return None
    if abtestid not in abtestids:
      return None

    dur = int(read_duration)
    if dur > 10000:
      dur = 120

    if dur <= 0:
      return None

    res = [abtestid, mid, docid, dur, rec_reason]
    return res

  if line.find("resp") == 0:
    if len(line.split("\t"))<44:
      return None
    (resp, mid, tm, article_cnt, index_num, mark, title, reason, read_num, topic, keywords, pub_time, image_type,img_list, url, account, channel, art_source, account_openid, abtestid, sub_topic, userinfo, position, app_ver,aduser_flag, location, pagetime, rec_reason, adid, vulgar, sub_list, ip, recall_word, video_type,channel_id,docid, product, source_type,distribution,model,lda_topic,xid,gid,imei) = line.replace("\n","").split("\t")[0:44]
    if(product!="sgsapp"):
      return None
    if(channel_id!="1"):
      return None
    if art_source=='15':
      return None
    if abtestid not in abtestids:
      return None

    res = [abtestid, mid, docid, 0, rec_reason]
    return res

  return None

d = d.map(parse).filter(lambda x: x != None)

info = {}

def deal(d):
  d_show = d.filter(lambda x: x[3] == 0) 
  show_count = d_show.count()
  d_click = d.filter(lambda x: x[3] > 0)
  click_count = d_click.count()

  click_users = d_click.map(lambda x: x[1]).distinct().count()
  total_users = d_show.map(lambda x: x[1]).distinct().count()

  total_dur = d_click.map(lambda x: x[3]).sum()

  res = dict(
      ctr=click_users / show_count,
      read_ratio=click_users / total_users,
      read_dur1=total_dur / total_users,
      read_dur2=total_dur / click_users,
      show_count=show_count,
      click_count=click_count,
      total_dur=total_dur,
      total_users=total_users,
      click_users=click_users
    )
  return res


for abtestid in abtestids:
  name = abtestid
  d = d.filter(lambda x: x[0] == abtestid)
  info[abtestid] = deal(d)
  print(name, info[name])

  name = 'cold:' + abtestid
  d_cold = d.filter(lambda x: x[4] in cold_reasons)
  info[name] = deal(d_cold)
  print(name, info[name])

  name = 'noncold:' + abtestid
  d_nocold = d.filter(lambda x: x[4] not in cold_reasons)
  info[name] = deal(d_nocold)
  print(name, info[name])

print(info)
