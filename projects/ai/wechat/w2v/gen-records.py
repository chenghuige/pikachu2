from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')

from absl import app, flags
FLAGS = flags.FLAGS

import tensorflow as tf
import tensorflow_addons as tfa
import glob
import numpy as np
import random
from multiprocessing import Pool, Manager, cpu_count
import pandas as pd
from collections import defaultdict
import pymp
from icecream import ic

import melt
import gezi
from gezi import tqdm

vocabs = {}
sentences = []

def build_features(index):
  out_dir = f'../input/{FLAGS.records_name}/{FLAGS.day}/{FLAGS.attr}/{FLAGS.window_size}'
  gezi.try_mkdir(out_dir)
  if index == 0:
    print(out_dir)
  ofile = f'{out_dir}/{index}.tfrec'

  total = len(sentences)
  start, end = gezi.get_fold(total, FLAGS.num_records, index)
  seqs = sentences[start:end]

  vocab = vocabs[FLAGS.attr]
  # Build the sampling table for vocab_size tokens.
  sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab.size())

  buffer_size = 10000 
  with melt.tfrecords.Writer(ofile, buffer_size=buffer_size, shuffle=True) as writer:
    # Iterate over all sequences (sentences) in dataset.
    for sequence in tqdm(seqs, desc=f'generate skip-grams, index:{index}'):
      # Generate positive skip-gram pairs for a sequence (sentence).
      positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            sequence,
            vocabulary_size=vocab.size(),
            sampling_table=sampling_table,
            window_size=FLAGS.window_size,
            negative_samples=0)

      for target_word, context_word in positive_skip_grams:
        fe = {}
        fe['target'] = target_word
        fe['pos'] = context_word
        writer.write_feature(fe)

def main(data_dir):
  np.random.seed(FLAGS.seed_)
  random.seed(FLAGS.seed_)

  FLAGS.corpus = FLAGS.corpus or f'../input/{FLAGS.day}/{FLAGS.attr}_corpus.txt'
  vocabs[FLAGS.attr] = gezi.Vocab(f'../input/{FLAGS.attr}_vocab.txt')

  vocab = vocabs[FLAGS.attr]

  with open(FLAGS.corpus) as f: 
    lines = f.read().splitlines()

  for line in tqdm(lines, desc='map id'):
    l = [vocab.id(x) for x in line.split()]
    sentences.append(l)

  ic(len(sentences))

  with gezi.Timer('shuffle sentences'):
    random.shuffle(sentences)

  if not FLAGS.num_records:
    FLAGS.num_records = cpu_count()

  with gezi.Timer('build_feature'):
    if FLAGS.debug:
      FLAGS.records_name = 'tfrecords.debug'
      build_features(0)
    elif FLAGS.index is not None:
      build_features(FLAGS.index)
    else:
      with Pool(FLAGS.num_records) as p:
        p.map(build_features, range(FLAGS.num_records))

if __name__ == '__main__':
  flags.DEFINE_integer('num_records', None, '')
  flags.DEFINE_integer('seed_', 12345, '')
  flags.DEFINE_string('records_name', 'tfrecords/w2v', '')
  flags.DEFINE_integer('index', None, '')

  flags.DEFINE_string('corpus', None, '')
  flags.DEFINE_string('day', '14.5', '')
  flags.DEFINE_string('attr', 'doc', '')
  flags.DEFINE_integer('window_size', 8, '')
  
  app.run(main) 
