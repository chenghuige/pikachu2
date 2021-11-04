#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   dataset.py
#        \author   chenghuige  
#          \date   2021-01-09 17:51:11.308942
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

from absl import app, flags
FLAGS = flags.FLAGS

import numpy as np
## TODO RuntimeError: Already borrowed for encode ... only can use BertTokenizer 
## https://github.com/huggingface/tokenizers/issues/537 not solve
# from transformers import AutoTokenizer as Tokenizer
from transformers import BertTokenizerFast as Tokenizer
# from transformers import BertTokenizer as Tokenizer
from sklearn.preprocessing import MultiLabelBinarizer
import jieba
import sentencepiece as spm

import tensorflow as tf
from tensorflow.keras import backend as K

import melt as mt
from qqbrowser.config import *
from qqbrowser import util
# from qqbrowser import util

sp = None

def cut(sentence):
  if FLAGS.segmentor == 'jieba':
    return jieba.cut(sentence)
  else:
    return sp.encode(sentence)

def encode_frames(frames):
  frames = frames.numpy() 
  frames_len = len(frames)
  num_frames = min(frames_len, FLAGS.max_frames)
  num_frames = np.array([num_frames], dtype=np.int32)
  return [frames[min(i, frames_len - 1)] for i in range(FLAGS.max_frames)], num_frames

def parse_frames(frames):
  # frames = tf.sparse.to_dense(frames)
  frames, num_frames = tf.py_function(encode_frames, [frames], [tf.string, tf.int32])
  frames_embedding = tf.map_fn(lambda x: tf.io.decode_raw(x, out_type=tf.float16), frames, dtype=tf.float16)
  if not FLAGS.fp16:
    frames_embedding = tf.cast(frames_embedding, tf.float32)
  frames_embedding.set_shape([FLAGS.max_frames, FLAGS.frame_embedding_size])
  num_frames.set_shape([1])
  return frames_embedding, num_frames

class Dataset(mt.Dataset):
  def __init__(self, subset='valid', **kwargs):
    super(Dataset, self).__init__(subset, **kwargs)
    if FLAGS.parse_strategy == 1:
      assert not FLAGS.batch_parse, 'frames decode must before batch, set --batch_parse=0'

    self.tokenizer = Tokenizer.from_pretrained(FLAGS.transformer)
    self.tag_vocab = gezi.Vocab('../input/tag_vocab.txt')
    self.word_vocab = gezi.Vocab('../input/word_vocab.txt', FLAGS.reserve_vocab_size)
    if FLAGS.parse_strategy <= 2:
      if FLAGS.label_strategy == 'selected_tags':
        self.selected_tags = set()
        with open(FLAGS.multi_label_file, encoding='utf-8') as fh:
          for line in fh:
            tag_id = int(line.strip())
            if FLAGS.parse_strategy > 1:
              tag_id = self.tag_vocab.id(tag_id)
            self.selected_tags.add(tag_id)
        self.num_labels = len(self.selected_tags)
        # ic(self.num_labels)
        assert self.num_labels == FLAGS.num_labels
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([self.selected_tags])   
    
    if FLAGS.segmentor == 'sp':
      global sp
      sp = spm.SentencePieceProcessor(model_file='../input/sp10w.model')
        
  def encode_text(self, text, max_len, last_tokens=0):
    text = text.numpy().decode(encoding='utf-8')
    #input_ids = self.tokenizer.encode(text)
    ## notice here not CLS SEP
    #tokens = self.tokenizer.tokenize(text)     
    #input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
    words = list(cut(text))

    if FLAGS.segmentor == 'jieba':
      words = [x for x in words if x.strip()]
      if not FLAGS.mix_segment:
        word_ids = [self.word_vocab.id(x) for x in words]
      else:
        word_ids = []
        for x in words:
          word_ids.extend(self.word_vocab.ids(x, FLAGS.vocab_size))
    else:
      word_ids = words
    word_ids = [101, *word_ids, 102]
    #tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
    res = self.tokenizer(text, return_offsets_mapping=True)
    input_ids = res.input_ids
    offsets = res.offset_mapping
    tokens = [text[s:e] for s,e in offsets]
    tokens[0] = '[CLS]'
    tokens[-1] = '[SEP]'

    input_ids = gezi.pad(input_ids, max_len, 0, last_tokens=last_tokens)

    if FLAGS.mask_words and FLAGS.segmentor == 'jieba':
      try:
        words_mask = gezi.words_mask([tokens[0], *words, tokens[-1]], tokens)
      except Exception:
        ic(text)
        exit(0)

      assert len(input_ids) == len(words_mask), text
      words_mask = gezi.pad(words_mask, max_len, 0, last_tokens=last_tokens)
      words_mask = gezi.fix_words_mask(words_mask)
    else:
      words_mask = input_ids

    mask = [int(x > 0) for x in input_ids]
    
    max_len_ = int(max_len * (2/3.))
    word_ids = gezi.pad(word_ids, max_len_, 0, last_tokens=10)
    return input_ids, mask, words_mask, word_ids

  def encode_text2(self, title, asr, max_title_len, max_len, max_words, last_title_tokens, last_asr_tokens):
    title = title.numpy().decode(encoding='utf-8')
    asr = asr.numpy().decode(encoding='utf-8')

    title_ids = self.tokenizer(title).input_ids
    asr_ids = self.tokenizer(asr).input_ids

    title_ids = gezi.reduce_len(title_ids, max_title_len, 0, last_tokens=last_title_tokens)
    input_ids = [*title_ids, *asr_ids[1:]]
    input_ids = gezi.pad(input_ids, max_len, 0, last_tokens=last_asr_tokens)

    attention_mask = [int(x > 0) for x in input_ids]

    token_type_ids = [0] * len(title_ids) + [1] * (len(input_ids) - len(title_ids))

    if FLAGS.segmentor == 'jieba':
      title_words = [x for x in cut(title) if x.strip()]
      asr_words = [x for x in cut(asr) if x .strip()]
      title_ids = [self.word_vocab.id(x) for x in title_words]
      asr_ids = [self.word_vocab.id(x) for x in asr_words]
    else:
      title_ids, asr_ids = cut(title), cut(asr_words)
    word_ids = [101, *title_ids, 102, *asr_ids, 102]

    word_ids = gezi.pad(word_ids, max_words, 0, last_tokens=10)

    return input_ids, attention_mask, token_type_ids, word_ids

  def parse_text(self, text, max_len, last_tokens=0):
    encode_text = lambda text: self.encode_text(text, max_len, last_tokens)
    # NOTICE! 特别注意！ 输入给pyfunc的输入变量 不能作为输出的变量。。。 比如 
    # tf.py_function(self.encode_text, [text, max_len, last_tokens], [tf.int32, tf.int32]) 
    # wrong!!!
    # 内部长度输出最终依赖max_len 这个 是不行的 可能是图模式原因 必须固定住！
    input_ids, mask, words_mask, word_ids = tf.py_function(encode_text, [text], [tf.int32, tf.int32, tf.int32, tf.int32])
    input_ids.set_shape([max_len])
    mask.set_shape([max_len])
    words_mask.set_shape([max_len])
    max_len_ = int(max_len * (2/3.))
    word_ids.set_shape([max_len_])
    return input_ids, mask, words_mask, word_ids

  def parse_text2(self, title, asr, max_title_len, max_len, max_words, last_title_tokens, last_asr_tokens):
    encode_text = lambda title, asr: self.encode_text2(title, asr, max_title_len, max_len, max_words, last_title_tokens, last_asr_tokens)
    # NOTICE! 特别注意！ 输入给pyfunc的输入变量 不能作为输出的变量。。。 比如 
    # tf.py_function(self.encode_text, [text, max_len, last_tokens], [tf.int32, tf.int32]) 
    # wrong!!!
    # 内部长度输出最终依赖max_len 这个 是不行的 可能是图模式原因 必须固定住！
    input_ids, attention_mask, token_type_ids, word_ids = tf.py_function(encode_text, [title, asr], [tf.int32, tf.int32, tf.int32, tf.int32])
    input_ids.set_shape([max_len])
    attention_mask.set_shape([max_len])
    token_type_ids.set_shape([max_len])
    word_ids.set_shape([max_words])
    return input_ids, attention_mask, token_type_ids, word_ids
  
  def parse_label(self, labels):
    tags = labels.numpy()
    # tag filtering
    tags = [tag for tag in tags if tag in self.selected_tags]
    multi_hot = self.mlb.transform([tags])[0].astype(dtype=np.int8)
    return tf.convert_to_tensor(multi_hot)
  
  def parse_label2(self, labels):
    tags = labels.numpy()
    tags = [self.tag_vocab.id(tag, 0) for tag in tags]
    if FLAGS.num_negs:
      negs, _, _ = tf.random.log_uniform_candidate_sampler(
        true_classes=[x - 2 for x in tags], 
        num_true=len(tags),  
        num_sampled=FLAGS.num_negs, 
        unique=True, 
        range_max=self.tag_vocab.size() - 2,  
        seed=FLAGS.seed, 
        name="negative_sampling"  # name of this operation
      )
      negs = tf.cast(negs, tf.int32)
    tags = gezi.pad(tags, FLAGS.max_tags)
    if FLAGS.num_negs:
      negs = [x + 2 for x in negs]
      tags.extend(negs)
    tags = tf.convert_to_tensor(tags)
    return tags
  
  def parse_labels3(self, labels):
    negs, _, _ = tf.random.log_uniform_candidate_sampler(
        true_classes=tf.cast(tf.reshape(labels, [-1, FLAGS.max_tags]), tf.int64) - 2,  # class that should be sampled as 'positive'
        num_true=FLAGS.max_tags,  # each positive skip-gram has 1 positive context class
        num_sampled=FLAGS.num_negs,  # number of negative context words to sample
        unique=True,  # all the negative samples should be unique
        range_max=self.tag_vocab.size() - 2,  # pick index of the samples from [0, vocab_size]
        seed=FLAGS.seed,  # seed for reproducibility
        name="negative_sampling"  # name of this operation
      )
    negs = tf.cast(negs, labels.dtype)
    negs += 2
    tags = tf.concat([tf.reshape(labels, (-1, FLAGS.max_tags)), 
                      tf.reshape(negs, (-1, FLAGS.num_negs))], -1)
    tags = tf.squeeze(tags, 0)
    return tags

  def parse_labels(self, labels):
    ## already done in mt.dataset decode
    # labels = tf.sparse.to_dense(labels)
    if FLAGS.label_strategy == 'selected_tags':
      labels = tf.py_function(self.parse_label, [labels], [tf.int8])[0]
      labels.set_shape([self.num_labels])
    elif FLAGS.label_strategy == 'all_tags':
      if FLAGS.use_pyfunc:
        labels = tf.py_function(self.parse_label2, [labels], [tf.int32])[0]
        labels.set_shape([FLAGS.max_tags + FLAGS.num_negs])
      else:
        labels = self.parse_labels3(labels)
    return labels
  
  def parse1(self, example):
    keys, excl_keys = [], []
    
    keys = ['id', 'category_id']
    if FLAGS.use_title:
      keys.append('title')
    if FLAGS.use_asr:
      keys.append('asr_text')
 
    varlen_keys = ['tag_id']
    if FLAGS.use_frames:
      varlen_keys.append('frame_feature')

    self.auto_parse(keys=keys, exclude_keys=excl_keys + varlen_keys)
    self.adds(varlen_keys)

    fe = self.parse_(serialized=example)
    
    if 'category_id'  in fe:
      mask = tf.cast(fe['category_id'] == -1, tf.int32)
      fe['cat'] = tf.cast(fe['category_id'] // 100, tf.int32) * (1 - mask)
      fe['subcat'] = tf.cast(fe['category_id'], tf.int32) * (1 - mask)
      del fe['category_id']
    else:
      fe['cat'] = tf.zeros_like(fe['id'], dtype=tf.int32)
      fe['subcat'] = fe['cat']
    
    fe['frames'], fe['num_frames'] = parse_frames(fe['frame_feature'])
    del fe['frame_feature']

    if not FLAGS.merge_text:
      fe['title_ids'], fe['title_mask'], fe['title_words_mask'], fe['title_word_ids'] = self.parse_text(fe['title'], FLAGS.max_title_len, FLAGS.last_title_tokens)
      
      if FLAGS.use_asr:
        fe['asr_ids'], fe['asr_mask'], fe['asr_words_mask'], fe['asr_word_ids'] = self.parse_text(fe['asr_text'], FLAGS.max_asr_len, FLAGS.last_asr_tokens)
    else:
      fe['input_ids'], fe['attention_mask'], fe['token_type_ids'], fe['word_ids'] = self.parse_text2(fe['title'], fe['asr_text'], 
                FLAGS.max_title_len, FLAGS.max_text_len, FLAGS.max_words,
                FLAGS.last_title_tokens, FLAGS.last_asr_tokens)

    del fe['title']
    del fe['asr_text']

    if not FLAGS.mask_words:
      del fe['title_words_mask']
      del fe['asr_words_mask']

    mt.try_append_dim(fe)
    
    x = fe
    
    x['vid'] = x['id']
    del x['id']

    if 'tag_id' in x:
      if FLAGS.dump_records:
        y = tf.zeros_like(x['vid'], dtype=tf.int32)
      else:
        if 'tags' in FLAGS.label_strategy:
          y = self.parse_labels(x['tag_id'])
        else:
          y = x['tag_id']
        del x['tag_id']
    else:    
      y = tf.zeros_like(x['vid'], dtype=tf.int32)
    
    return x, y
  
  def parse_x(self, x):
    x['frames'] = tf.reshape(x['frames'], [FLAGS.max_frames, -1])

    if FLAGS.asr_len:
      x['asr_ids'], x['asr_mask'] = x['asr_ids'][:FLAGS.asr_len], x['asr_mask'][:FLAGS.asr_len]

    if 'tag_id' in x:
      if 'tags' in FLAGS.label_strategy:
        x['tag_id'] = self.parse_labels(x['tag_id'])
        # tf.print(x['tag_id'].shape)
        if FLAGS.label_strategy == 'all_tags':
          x['pos'] = x['tag_id'][:FLAGS.max_tags]
          x['neg'] = x['tag_id'][FLAGS.max_tags:]
          # tf.print(x['pos'].shape, x['neg'].shape)
        if FLAGS.label_strategy == 'selected_tags':
          y = x['tag_id']
        elif FLAGS.label_strategy == 'all_tags':
          if FLAGS.tag_pooling:
            y = tf.zeros_like(x['tag_id'][0], dtype=mt.get_float())
          else:
            if 'pos' not in x:
              y = x['tag_id']
            else:
              # ic(x['tag_id'].shape, x['pos'].shape, x['neg'].shape)
              y = tf.concat([tf.ones_like(x['pos']), tf.zeros_like(x['neg'])], -1)

      else:
        y = x['tag_id']
      if not FLAGS.dump_records:
        del x['tag_id']
    else:    
      # if FLAGS.tag_pooling:
      y = tf.zeros_like(x['vid'][0], dtype=mt.get_float())

    x['y'] = y
    return x, y 
  
  def parse2(self, example):
    keys, excl_keys, varlen_keys = [], [], []
    self.auto_parse(keys=keys, exclude_keys=excl_keys + varlen_keys)
    self.adds(varlen_keys)

    fe = self.parse_(serialized=example)
    mt.try_append_dim(fe)
    x = fe    
    
    return self.parse_x(x)
  
   # this is for pairwise parse
  def parse3(self, example):
    if self.subset == 'test':
      return self.parse2(example)
    else:
      x = self.basic_parse(example)
      x1, x2 = util.split_pairwise(x)
      x1, _ = self.parse_x(x1)
      x2, _ = self.parse_x(x2)
      # 目前似乎dict嵌套有问题 TODO 
      # 让输出能找到vid1, vid2内容
      x1['vid1'] = x1['vid']
      x2['vid2'] = x2['vid']
      if self.subset == 'train':
        return (x1, x2), x[FLAGS.relevance]
      return (x1, x2), x['relevance']
      
  def parse(self, example):
    parse_fn = getattr(self, f'parse{FLAGS.parse_strategy}')
    return parse_fn(example)
  
