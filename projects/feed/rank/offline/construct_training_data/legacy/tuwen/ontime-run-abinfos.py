#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   run.py
#        \author   chenghuige  
#          \date   2015-02-26 17:48:25.559868
#   \Description  
# ==============================================================================

import sys,os
import time,datetime

minutes = range(6) 
minutes = set([x * 10 + 9 for x in minutes])

pre = None
while(True):
  now = time.strftime('%Y%m%d',time.localtime(time.time())) 
  l = time.localtime()
  now_min = int(l[4])
  now_sec = int(l[5])
  if now_min == 59 and now_sec == 0:
    print(l)
    command = 'sh ./abinfos-hour.sh >> /tmp/abinfos.log 2>&1 &'
    print('command:', command)
    os.system(command)
    time.sleep(2)
    
  #if now_min in minutes and now_sec == 0:
  #  print(l)
  #  command = 'sh ./abinfos.sh >> /tmp/abinfos.log 2>&1 &'
  #  print('command:', command)
  #  os.system(command)
  #  time.sleep(2)



 
