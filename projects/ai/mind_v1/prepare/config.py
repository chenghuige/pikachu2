#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   config.py
#        \author   chenghuige  
#          \date   2020-08-27 08:32:13.481856
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
FLAGS = flags.FLAGS

#flags.DEFINE_bool('record_padded', False, '')

flags.DEFINE_integer('max_history', 200, '')
flags.DEFINE_integer('max_lookup_history', 200, '')
flags.DEFINE_integer('max_impressions', 300, '')
flags.DEFINE_integer('max_titles', 50, '')
flags.DEFINE_integer('max_bert_titles', 50, '')
flags.DEFINE_integer('max_abstracts', 10, '50 is better')
flags.DEFINE_integer('max_bert_abstracts', 50, '')
flags.DEFINE_integer('max_bodies', 10, '50 is better')
flags.DEFINE_integer('max_bert_bodies', 50, '')

flags.DEFINE_integer('max_title_entities', 9, '')
flags.DEFINE_integer('max_abstract_entities', 30, '')

flags.DEFINE_integer('max_his_title_entities', 2, '')
flags.DEFINE_integer('max_his_abstract_entities', 2, '')
