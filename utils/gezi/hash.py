#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   hash.py
#        \author   chenghuige  
#          \date   2018-04-28 12:04:55.557328
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import six 
import numpy as np

import hashlib
try:
  import mmh3
  def hash_str(key):
    if not six.PY2:
      return hex(mmh3.hash128(key.encode('utf8')) % sys.maxsize)[2:]
    else:
      # ignore last L
      return hex(mmh3.hash128(key.encode('utf8')) % sys.maxsize)[2:-1]
except Exception:
  def hash_str(key):
    if not six.PY2:
      return hex(int(hashlib.sha256(key.encode('utf8')).hexdigest(), 16) % sys.maxsize)[2:]
    else:
      # ignore last L
      return hex(int(hashlib.sha256(key.encode('utf8')).hexdigest(), 16) % sys.maxsize)[2:-1]

# NOTICE ! fasttext use uint32_t !
def fasttext_hash(word):
  h = 2166136261
  for w in word:
    h = np.uint32(h ^ ord(w))
    h = np.uint32(h * 16777619)
  return h

hash = fasttext_hash

def feature_hash(feature, dim, seed=123):
    """Feature hashing.

    Args:
        feature (str): Target feature represented as string.
        dim (int): Number of dimensions for a hash value.
        seed (float): Seed of a MurmurHash3 hash function.

    Returns:
        numpy 1d array: one-hot-encoded feature vector for `s`.

    """
    vec = np.zeros(dim)
    i = mmh3.hash(feature, seed) % dim
    vec[i] = 1
    return vec 
  
# def hash(key, maxsize=2**64):
#   return mmh3.hash128(key) % maxsize

# def hash_uint64(key, maxsize=2**64):
#   return hash(key, maxsize)

# def hash_int64(key, maxsize=2**64):
#   return hash(key, maxsize) - 2**63 

def hash(key):
  return mmh3.hash64(key, signed=False)[0] 

def hash_uint64(key):
  key = str(key)
  return mmh3.hash64(key, signed=False)[0]

def hash_int64(key):
  key = str(key)
  # return mmh3.hash64(key)[0] % maxsize - 2**63
  return mmh3.hash64(key)[0]
  # return int(int(hashlib.sha256(key.encode('utf8')).hexdigest(), 16) % maxsize)

# https://github.com/YannDubs/Hash-Embeddings/blob/master/hashembed/embedding.py
class HashFamily():
    r"""Universal hash family as proposed by Carter and Wegman.
    .. math::
            \begin{array}{ll}
            h_{{a,b}}(x)=((ax+b)~{\bmod  ~}p)~{\bmod  ~}m \ \mid p > m\\
            \end{array}
    Args:
        bins (int): Number of bins to hash to. Better if a prime number.
        mask_zero (bool, optional): Whether the 0 input is a special "padding" value to mask out.
        moduler (int,optional): Temporary hashing. Has to be a prime number.
    """

    def __init__(self, bins, mask_zero=True, moduler=None):
        if moduler and moduler <= bins:
            raise ValueError("p (moduler) should be >> m (buckets)")

        self.bins = bins
        self.moduler = moduler if moduler else self._next_prime(np.random.randint(self.bins + 1, 2**32))
        self.mask_zero = mask_zero

        # do not allow same a and b, as it could mean shifted hashes
        self.sampled_a = set()
        self.sampled_b = set()

    def _is_prime(self, x):
        """Naive is prime test."""
        for i in range(2, int(np.sqrt(x))):
            if x % i == 0:
                return False
        return True

    def _next_prime(self, n):
        """Naively gets the next prime larger than n."""
        while not self._is_prime(n):
            n += 1

        return n

    def draw_hash(self, a=None, b=None):
        """Draws a single hash function from the family."""
        if a is None:
            while a is None or a in self.sampled_a:
                a = np.random.randint(1, self.moduler - 1)
                assert len(self.sampled_a) < self.moduler - 2, "please give a bigger moduler"

            self.sampled_a.add(a)
        if b is None:
            while b is None or b in self.sampled_b:
                b = np.random.randint(0, self.moduler - 1)
                assert len(self.sampled_b) < self.moduler - 1, "please give a bigger moduler"

            self.sampled_b.add(b)

        if self.mask_zero:
            # The return doesn't set 0 to 0 because that's taken into account in the hash embedding
            # if want to use for an integer then should uncomment second line !!!!!!!!!!!!!!!!!!!!!
            return lambda x: ((a * x + b) % self.moduler) % (self.bins - 1) + 1
            # return lambda x: 0 if x == 0 else ((a*x + b) % self.moduler) % (self.bins-1) + 1
        else:
            return lambda x: ((a * x + b) % self.moduler) % self.bins

    def draw_hashes(self, n, **kwargs):
        """Draws n hash function from the family."""
        return [self.draw_hash() for i in range(n)]

