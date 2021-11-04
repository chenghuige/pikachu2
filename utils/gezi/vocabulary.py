# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import mmh3
from absl import logging

"""copy from google im2txt tensorflow/models/im2txt/inference_util, 
make it possible to replace c++ Version Vocabulary,
add num_reserved_ids"""

def hash(key):
  return mmh3.hash64(key, signed=False)[0] 

class Vocabulary(object):
  """Vocabulary class for an image-to-text model."""

  def __init__(self,
               vocab_file=None,
               num_reserved_ids=1,
               start_word="<S>",
               end_word="</S>",
               unk_word="<UNK>",
               pad_word="<PAD>",
               min_count=None,
               max_words=None,
               buckets=None,
               fixed=False,
               simple=False,
               append=False):
    """Initializes the vocabulary.

    Args:
      vocab_file: File containing the vocabulary, where the words are the first
        whitespace-separated token on each line (other tokens are ignored) and
        the word ids are the corresponding line numbers.
      num_reserved_ids: might be 1 for vocab embedding make 0 as <PAD>
      start_word: Special word denoting sentence start.
      end_word: Special word denoting sentence end.
      unk_word: Special word denoting unknown words.

      TODO add count support vocab.count(i) vocab.count(word)
    """
    # HACK!
    if vocab_file and 'ngram' in vocab_file:
      if start_word == '<S>':
        start_word = '<<S>>'
        end_word = '<</S>>'
        unk_word = '<<UNK>>'
        pad_word = '<<PAD>>'
        
    if vocab_file and (not buckets):
      if vocab_file and not os.path.exists(vocab_file):
        logging.fatal("Vocab file %s not found.", vocab_file)
      logging.debug("Initializing vocabulary from file: %s", vocab_file)

      with open(vocab_file, encoding="utf8", mode="r") as f:
        lines = list(f.readlines())
        lines = [line.rstrip('\n').split('\t') for line in lines]
 
      if min_count:
        reverse_vocab = [x for x,y in lines if int(y) >= min_count]
      else:
        reverse_vocab = [x[0] if isinstance(x, (tuple, list)) else x for x in lines]

      self.counts = []
      if len(lines[0]) == 2:
        try:
          self.counts = [int(y) for x,y in lines]
        except Exception:
          pass

      if max_words:
        reverse_vocab = reverse_vocab[:max_words]

      if fixed:
        num_reserved_ids = 0

      reserved_vocab = []
      if num_reserved_ids > 0:
        reserved_vocab = [pad_word] * num_reserved_ids

      self.ori_size = len(reverse_vocab) 
      #print('----re', reverse_vocab)

      reserved_vocab2 = []
      if not fixed and not simple:
        if unk_word not in reverse_vocab and unk_word.lower() not in reverse_vocab:
          reserved_vocab2.append(unk_word)
        if start_word and start_word not in reverse_vocab and start_word.lower() not in reverse_vocab:
          reserved_vocab2.append(start_word)
        if end_word and end_word not in reverse_vocab and end_word.lower() not in reverse_vocab:
          reserved_vocab2.append(end_word)

      if append:
        # old version append mode
        reverse_vocab = reserved_vocab + reverse_vocab + reserved_vocab2
        self.num_insert_words = len(reserved_vocab)
        self.num_append_words = len(reserved_vocab2)
      else:
        # new version not append but insert before
        reverse_vocab = reserved_vocab + reserved_vocab2 + reverse_vocab
        self.num_insert_words = len(reserved_vocab) + len(reserved_vocab2)
        self.num_append_words = 0

      vocab = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])

      if self.counts:
        max_counts = self.counts[0]
        max_counts += 1
        self.counts = [max_counts] * self.num_insert_words + self.counts + [1] * self.num_append_words

        # print(len(self.counts), self.num_insert_words, self.num_append_words)

      logging.debug("Created vocabulary with %d words" % len(vocab))

      self.vocab = vocab  # vocab[word] = id
      self.reverse_vocab = reverse_vocab  # reverse_vocab[id] = word
    else:
      logging.debug('Vocabulary just using hash with buckets', buckets)

      reserved_vocab = []
      reverse_vocab = []
      if num_reserved_ids > 0:
        reserved_vocab = [pad_word] * num_reserved_ids    
        reserved_vocab2 = [unk_word, start_word, end_word]
        reverse_vocab = reserved_vocab + reserved_vocab2 

      vocab = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])

      self.vocab = vocab  # vocab[word] = id
      self.reverse_vocab = reverse_vocab  # reverse_vocab[id] = word

    # Save special word ids.
    if not fixed and not simple:
      if start_word:
        self._start_id = vocab[start_word]
      if end_word:
        self._end_id = vocab[end_word] 

    if unk_word in vocab:
      self._unk_id = vocab[unk_word] 
    else:
      self._unk_id = 1

    self._buckets = buckets
    self.num_reserved_ids = num_reserved_ids

    self.start_word = start_word
    self.end_word = end_word
    self.unk_word = unk_word

  def is_special(self, word):
    return word == self.unk_word or word == self.start_word or word == self.end_word
    
  def word_to_id(self, word):
    """Returns the integer word id of a word string."""
    if word in self.vocab:
      return self.vocab[word]
    else:
      if not self._buckets:
        return self._unk_id
      else:
        return self.size() + hash(word) % self._buckets

  def id(self, word, default=None):
    """Returns the integer word id of a word string."""
    word = str(word)
    if word in self.vocab:
      return self.vocab[word]
    else:
      if default is not None:
        return default
      if not self._buckets:
        return self._unk_id
      else:
        return self.size() + hash(word) % self._buckets
  
  def __call__(self, word, default=None):
    return self.id(word, default)

  def id_to_word(self, word_id):
    """Returns the word string of an integer word id."""
    if word_id >= len(self.reverse_vocab):
      return self.reverse_vocab[self._unk_id]
    else:
      return self.reverse_vocab[word_id]

  def key(self, word_id):
    """Returns the word string of an integer word id."""
    if word_id >= len(self.reverse_vocab):
      return self.reverse_vocab[self._unk_id]
    else:
      return self.reverse_vocab[word_id]
    
  def __getitem__(self, word_id):
    return self.key(word_id)

  def count(self, word_id):
    if not self.counts:
      return 1 
    else:
      return self.counts[word_id]

  def count_word(self, word):
    id_ = self.id(word)
    # if id_ == self._unk_id:
    #   return 0
    return self.count(id_)

  def count_key(self, word):
    id_ = self.id(word)
    # if id_ == self._unk_id:
    #   return 0
    return self.count(id_)

  def size(self, min_count=0):
    if not min_count or not self.counts:
      return len(self.reverse_vocab)
    else:
      for i in range(len(self.counts)):
        if self.counts[i] < min_count:
          return i
      return len(self.counts)

  def start_id(self):
    return self._start_id 

  def end_id(self):
    return self._end_id

  def unk_id(self):
    return self._unk_id

  #TODO how to use 'if word in vocab' ?
  def has(self, word):
    return word in self.vocab

  def add(self, word):
    if not word in self.vocab:
      id = len(self.reverse_vocab)
      self.vocab[word] = id
      self.reverse_vocab.append(word)

  def words(self):
    return self.reverse_vocab[self.num_reserved_ids:]

  def save(self, vocab_vile):
    with open(vocab_vile, 'w') as f: 
      for word in self.words():
        print(word, file=f)

  def ids(self, word, vocab_size=None):
    id_ = self.id(word)
    if id_ != self._unk_id and (not vocab_size or id_ < vocab_size):
      return [id_]
    else:
      l = []
      for ch in word:
        id_ = self.id(ch)
        if id_ < vocab_size:
          l.append(id_)
        else:
          l.append(self._unk_id)
      return l

class Vocab(Vocabulary):
  def __init__(self,
               vocab_file=None,
               num_reserved_ids=1,
               min_count=None,
               max_words=None,
               **kwargs):
    super(Vocab, self).__init__(vocab_file, num_reserved_ids, min_count=min_count, max_words=max_words, start_word=None, end_word=None, **kwargs)
