#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   write-tb.py
#        \author   chenghuige  
#          \date   2019-11-21 00:41:11.847855
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import traceback  

import gezi
from gezi import SummaryWriter

root_dir = sys.argv[1]
input = sys.stdin

os.system('mkdir -p %s' % root_dir)

if len(sys.argv) > 2:
  input = [x.strip() for x in open(sys.argv[2]).readlines()]

limit = 0
if len(sys.argv) > 3:
  limit = int(sys.argv[3])
  input = input[-limit:]

keys = ['gold/auc', 'group/auc', 'group/click/time_auc', 'auc', 'click/time_auc']
onlines = ['off', 'on']

mark = None
is_online = False
testid = None
testid_ = None

writers = {}

tags = set()

for line in input: 
  try:
    l = line.split('\t')
    if len(l) != 4:
      continue
    name, hour, mark, info = l
    tag = '\t'.join(l[:3])
    if tag in tags:
      continue 
    else:
      tags.add(tag)
    
    if ',' in name:
      testid = 'base'
    else:
      testid = None

    if 'online' in name:
      is_online = True
    else:
      is_online = False
    
    if not testid:
      testid = name.split('abid')[-1]
      testid_ = testid
    
    if not 'inverse_ratio' in name: 
      dir = root_dir + '/' + mark + '/' + onlines[int(is_online)] + '/' +  testid
      if not is_online and testid is 'base':
        if not testid_:
          continue
        dir = dir + '/' + testid_
    else:
      dir = root_dir + '/' + mark + '/' + 'inverse' + '/' + testid 
 
    if dir not in writers:
      writers[dir] = SummaryWriter(dir)
      if limit:
        command = 'rm -rf %s/*' % dir
        print(command)
        os.system(command)
    logger = writers[dir]
    
    step_file = os.path.join(dir, 'step.txt')
    if not os.path.exists(step_file):
      os.system('touch %s' % step_file)
      step = 1
    else:
      step = gezi.read_int_from(step_file, 0)
      step += 1
    gezi.write_to_txt(step, step_file)

    if 'inverse_ratio' in name: 
      # if step == 1:
      #   logger.scalar('inverse_ratio', 0., 0)
      logger.scalar('inverse_ratio', float(info), step)
    else:
      infos = [x.split(':') for x in info.split() if not x.startswith('version')]
      for key, val in infos:
        logger.scalar(key, float(val), step)
        if key in keys:
          logger.scalar('AAA/%s' % key, float(val), step)
  except Exception:
    print(traceback.format_exc(), file=sys.stderr)
    pass


  


