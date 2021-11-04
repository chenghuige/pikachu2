#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   timer.py
#        \author   chenghuige  
#          \date   2016-08-15 16:32:21.015897
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
@TODO may be

with gezi.Timer('abc') as timer:
  ....
"""
import gezi 
logging = gezi.logging

import sys, time
class Timer():
  def __init__(self, info='', print_start=None, 
               print_fn=print, reset=None):
    self.start_time = time.time()
    if info and print_start:
      print_fn('%s start' % info)
    self.print_start = print_start
    self.info = info
    self.print_fn = print_fn
    self._reset = reset
    self.step = 0
    
  def __enter__(self):
    if self.info and self.print_start is None:
      self.print_fn('%s start' % self.info)
    return self

  def __exit__(self, type, value, trace):
    self.print_elapsed()

  def reset(self):
    self.start_time = time.time()

  def elapsed(self, reset=True):
    end_time = time.time()
    duration = end_time - self.start_time
    if reset and not (self._reset == False):
      self.start_time = end_time 
    return duration  

  def elapsed_minutes(self, reset=True):
    return self.elapsed(reset) / 60

  def elapsed_hours(self, reset=True):
    return self.elapsed(reset) / 3600

  def elapsed_ms(self, reset=True):
    return self.elapsed(reset) * 1000
  
  #ipython not allow this?.. FIXME
  def print(self):
    if self.info:
      self.print_fn('{} duration: {}'.format(self.info, self.elapsed()))
    else:
      self.print_fn(self.elapsed())

  def print_elapsed(self):
    if self.info:
      self.print_fn('{} duration: {}'.format(self.info, self.elapsed()))
    else:
      self.print_fn(self.elapsed())
      