#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   _stats.pyx
#        \author   chenghuige
#          \date   2019-09-02 12:31:18.222988
#   \Description
# ==============================================================================


from __future__ import absolute_import

from cpython cimport bool
from libc cimport math
cimport cython
cimport numpy as np
from numpy.math cimport PI
from numpy cimport ndarray, int64_t, float64_t, intp_t

import numpy as np
import scipy.stats
import scipy.special


@cython.wraparound(False)
@cython.boundscheck(False)
def _weighted_inverse_sort(intp_t[:] data, intp_t left, intp_t right, intp_t[:] temp, intp_t[:] weight_dis):

  if right - left < 1:
    return 0

  if right - left == 1:
    if data[left] <= data[right]:
      return 0
    else:
      data[left], data[right] = data[right], data[left]
      return data[right] - data[left]

  cdef:
    intp_t mid = 0
    intp_t dis_w = 0
    intp_t i = 0 
    intp_t j = 0 
    intp_t index = 0

  mid = (left + right) // 2
  dis_w = _weighted_inverse_sort(data, left, mid, temp, weight_dis) + _weighted_inverse_sort(data, mid + 1, right, temp, weight_dis)

  # print('----------')
  # print(left, right)
  # print(data)
  # print(temp)

  weight_dis[mid] = data[mid] 
  for i in reversed(range(left, mid)):
    weight_dis[i] = weight_dis[i + 1] + data[i]

  i = left
  j = mid + 1
  index = left

  while i <= mid and j <= right:
    if data[i] <= data[j]:
      temp[index] = data[i]
      i += 1
    else:
      temp[index] = data[j]
      dis_w += weight_dis[i] - (mid - i + 1) * data[j] 
      j += 1
    index += 1

  while i <= mid:
    temp[index] = data[i]
    i += 1
    index += 1

  while j <= right:
    temp[index] = data[j]
    j += 1
    index += 1

  i = left
  while i <= right:
    data[i] = temp[i]
    i += 1


  return dis_w
