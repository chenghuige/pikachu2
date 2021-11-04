#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-dict.py
#        \author   chenghuige  
#          \date   2020-08-21 14:12:10.754171
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
from tqdm import tqdm
import json

import gezi

files = [
  '../input/big/train/entity_embedding.vec',
  '../input/big/dev/entity_embedding.vec',
  '../input/big/test/entity_embedding.vec'
  ]

entity_vocab = gezi.WordCounter()

entities = set()

for file in files:
  for line in open(file):
    entity = line.strip().split()[0]
    if entity in entities:
      continue
    entities.add(entity)
    entity_vocab.add(entity)


cat_vocab = gezi.WordCounter()
scat_vocab = gezi.WordCounter()
entity_vocab = gezi.WordCounter()
entity_type_vocab = gezi.WordCounter()

  
files = [
  '../input/big/train/news.tsv',
  '../input/big/dev/news.tsv',
  '../input/big/test/news.tsv'
  ]

dids = set()

for file in files:
  total = len(open(file).readlines())
  for line in tqdm(open(file), total=total):
    l = line.strip('\n').split('\t')
    did = l[0]
    if did in dids:
      continue
    dids.add(did)
    cat, sub_cat = l[1], l[2]
    cat_vocab.add(cat)
    scat_vocab.add(sub_cat)

    title_entities = l[-2]
    abstract_entities = l[-1]

    title_entities = json.loads(title_entities)
    for m in title_entities:
      entity = m['WikidataId']
      entity_vocab.add(entity)
      entity_type_vocab.add(m['Type'])

    abstract_entities = json.loads(abstract_entities)
    for m in abstract_entities:
      entity = m['WikidataId']
      entity_vocab.add(entity)
      entity_type_vocab.add(m['Type'])


cat_vocab.save('../input/big/cat.txt')
scat_vocab.save('../input/big/sub_cat.txt')
entity_vocab.save('../input/big/entity.txt')
entity_type_vocab.save('../input/big/entity_type.txt')
