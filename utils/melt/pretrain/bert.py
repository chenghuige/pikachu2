#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   bert.py
#        \author   chenghuige
#          \date   2021-10-03 14:38:22.374010
#   \Description
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.python.ops.array_ops import sequence_mask
import melt as mt
import gezi

from transformers import TFAutoModel, TFAutoModelForMaskedLM, AutoConfig, BertConfig, TFBertModel
from transformers import BertTokenizerFast as Tokenizer

# https://www.kaggle.com/yihdarshieh/masked-my-dear-watson-mlm-with-tpu


def sample_without_replacement(prob_dist, nb_samples):
  """Sample integers in the range [0, N), without replacement, according to the probability
       distribution `prob_dist`, where `N = prob_dist.shape[0]`.
    
    Args:
        prob_dist: 1-D tf.float32 tensor.
    
    Returns:
        selected_indices: 1-D tf.int32 tensor
    """

  nb_candidates = tf.shape(prob_dist)[0]
  logits = tf.math.log(prob_dist)
  z = -tf.math.log(-tf.math.log(
      tf.random.uniform(shape=[nb_candidates], minval=0, maxval=1)))
  _, selected_indices = tf.math.top_k(logits + z, nb_samples)

  return selected_indices


def get_masked_lm_fn(tokenizer,
                     mlm_mask_prob=0.15,
                     mask_type_probs=(0.8, 0.1, 0.1),
                     token_counts=None,
                     predict_special_tokens=False,
                     mlm_smoothing=0.7):
  """
    Prepare the batch: from the input_ids and the lenghts, compute the attention mask and the masked label for MLM.

    Args:

        tokenizer: A Hugging Face tokenizer.  
        
        token_counts: A list of integers of length `tokenizer.vocab_size`, which is the token counting in a dataset
            (usually, the huge dataset used for pretraing a LM model). This is used for giving higher probability
            for rare tokens to be masked for prediction. If `None`, each token has the same probability to be masked.

        mlm_mask_prob:  A `tf.float32` scalar tensor. The probability to <mask> a token, inclding
            actually masking, keep it as it is (but to predict it), and randomly replaced by another token.
        
        mask_type_probs: A `tf.float32` tensor of shape [3]. Among the sampled tokens to be <masked>, 

            mask_type_probs[0]: the proportion to be replaced by the mask token
            mask_type_probs[1]: the proportion to be kept as it it
            mask_type_probs[2]: the proportion to be replaced by a random token in the tokenizer's vocabulary
        
        predict_special_tokens: bool, if to mask special tokens, like cls, sep or padding tokens. Default: `False`
        
        mlm_smoothing: float, smoothing parameter to emphasize more rare tokens (see `XLM` paper, similar to word2vec).
        
    Retruns:

        prepare_masked_lm_batch: a function that masks a batch of token sequences.
    """
  if token_counts is None:
    """
        Each token has the same probability to be masked.
        """
    token_counts = [1] * tokenizer.vocab_size

  # Tokens with higher counts will be masked less often.
  # If some token has count 1, it will have freq 1.0 in this frequency list, which is the highest value.
  # However, since it never appears in the corpus used for pretraining, there is no effect of this high frequency.

  token_mask_freq = np.maximum(token_counts, 1) ** -mlm_smoothing

  # NEVER to mask/predict padding tokens.
  token_mask_freq[tokenizer.pad_token_id] = 0.0

  if not predict_special_tokens:
    for special_token_id in tokenizer.all_special_ids:
      """
            Do not to predict special tokens, e.g. padding, cls, sep and mask tokens, etc.
            """
      token_mask_freq[special_token_id] = 0.0

  # Convert to tensor.
  token_mask_freq = tf.constant(token_mask_freq, dtype=tf.float32)

  mlm_mask_prob = tf.constant(mlm_mask_prob)
  mask_type_probs = tf.constant(mask_type_probs)

  vocab_size = tf.constant(tokenizer.vocab_size)
  pad_token_id = tf.constant(tokenizer.pad_token_id)
  mask_token_id = tf.constant(tokenizer.mask_token_id)

  def prepare_masked_lm_batch(inputs):
    """
        Prepare the batch: from the input_ids and the lenghts, compute the attention mask and the masked label for MLM.

        Args:
            
            inputs: a dictionary of tensors. Format is:
            
                {
                    'input_ids': `tf.int32` tensor of shape [batch_size, seq_len] 
                    : `tf.int32` tensor of shape [batch_size, seq_len] 
                }            
                
                Optionally, it could contain extra keys 'attention_mask' and `token_type_ids` with values being
                `tf.int32` tensors of shape [batch_size, seq_len] 
             
        Returns:
        
            result: a dictionary. Format is as following:

                {
                    'inputs': A dictionary of tensors, the same format as the argument `inputs`.
                    'mlm_labels': shape [batch_size, seq_len]
                    'mask_types': shape [batch_size, seq_len]
                    'original_input_ids': shape [batch_size, seq_len]
                    'nb_tokens': shape [batch_size]
                    'nb_non_padding_tokens': shape [batch_size]
                    'nb_tokens_considered': shape [batch_size]
                    'nb_tokens_masked': shape [batch_size]
                }
                
                The tensors associated to `number of tokens` are the toekn countings in the whole batch, not
                in individual examples. They are actually constants, but reshapped to [batch_size], because
                `tf.data.Dataset` requires the batch dimension to be consistent. These are used only for debugging,
                except 'nb_tokens_masked, which is used for calculating the MLM loss values.
        """

    input_ids = inputs['input_ids']

    # TODO batch_size might be None?
    batch_size, seq_len = input_ids.shape
    # batch_size, seq_len = mt.get_shape(input_ids, 0), mt.get_shape(input_ids, 1)
    # tf.print(batch_size)

    attention_mask = None
    if 'attention_mask' in inputs:
      attention_mask = inputs['attention_mask']

    # Compute `attention_mask` if necessary
    if attention_mask is None:
      attention_mask = tf.cast(input_ids != pad_token_id, tf.int32)

    # The number of tokens in each example, excluding the padding tokens.
    # shape = [batch_size]
    lengths = tf.reduce_sum(attention_mask, axis=-1)

    # The total number of tokens, excluding the padding tokens.
    nb_non_padding_tokens = tf.math.reduce_sum(lengths)

    # For each token in the batch, get its frequency to be masked from the 1-D tensor `token_mask_freq`.
    # We keep the output to remain 1-D, since it's easier for using sampling method `sample_without_replacement`.
    # shape = [batch_size * seq_len], 1-D tensor.
    freq_to_mask = tf.gather(params=token_mask_freq,
                             indices=tf.reshape(input_ids, [-1]))

    # Normalize the frequency to get a probability (of being masked) distribution over tokens in the batch.
    # shape = [batch_size * seq_len], 1-D tensor.
    prob_to_mask = freq_to_mask / tf.reduce_sum(freq_to_mask)

    tokens_considered = tf.cast(attention_mask, tf.bool)
    if not predict_special_tokens:
      for special_token_id in tokenizer.all_special_ids:
        tokens_considered = tf.logical_and(tokens_considered,
                                           input_ids != special_token_id)
    nb_tokens_considered = tf.reduce_sum(
        tf.cast(tokens_considered, dtype=tf.int32))

    # The number of tokens to be masked.
    # type = tf.float32
    # nb_tokens_to_mask = tf.math.ceil(mlm_mask_prob * tf.cast(nb_non_padding_tokens, dtype=tf.float32))
    nb_tokens_to_mask = tf.math.ceil(
        mlm_mask_prob * tf.cast(nb_tokens_considered, dtype=tf.float32))

    # round to an integer
    nb_tokens_to_mask = tf.cast(nb_tokens_to_mask, tf.int32)

    # Sample `nb_tokens_to_mask` of different indices in the range [0, batch_size * seq_len).
    # The sampling is according to the probability distribution `prob_to_mask`, without replacement.
    # shape = [nb_tokens_to_mask]
    indices_to_mask = sample_without_replacement(prob_to_mask,
                                                 nb_tokens_to_mask)

    # Create a tensor of shape [batch_size * seq_len].
    # At the indices specified in `indices_to_mask`, it has value 1. Otherwise, the value is 0.
    # This is a mask (after being reshaped to 2D tensor) for masking/prediction, where `1` means that, at that place,
    # the token should be masked for prediction.
    # (For `tf.scatter_nd`, check https://www.tensorflow.org/api_docs/python/tf/scatter_nd)
    pred_mask = tf.scatter_nd(
        indices=
        indices_to_mask[:, tf.
                        newaxis],  # This is necessary for making `tf.scatter_nd` work here. Check the documentation.
        updates=tf.cast(tf.ones_like(indices_to_mask), tf.bool),
        shape=[batch_size * seq_len]
        )

    # batch_size = -1
    # Change to 2-D tensor.
    # The mask for masking/prediction.
    # shape = [batch_size, seq_len]
    # pred_mask = tf.reshape(pred_mask, [batch_size, seq_len])
    pred_mask = tf.reshape(pred_mask, [batch_size, seq_len])

    # Get token ids at the places where to mask tokens.
    # 1-D tensor, shape = [nb_tokens_to_mask].
    _input_ids_real = input_ids[pred_mask]

    # randomly select token ids from the range [0, vocab_size)
    # 1-D tensor, shape = [nb_tokens_to_mask]
    _input_ids_rand = tf.random.uniform(shape=[nb_tokens_to_mask],
                                        minval=0,
                                        maxval=vocab_size,
                                        dtype=tf.int32)

    # A constant tensor with value `mask_token_id`.
    # 1-D tensor, shape = [nb_tokens_to_mask]
    _input_ids_mask = mask_token_id * tf.ones_like(_input_ids_real,
                                                   dtype=tf.int32)

    # For each token to be masked, we decide which type of transformations to apply:
    #     0: masked, 1: keep it as it is, 2: replaced by a random token

    # Detail: we need to pass log probability (logits) to `tf.random.categorical`,
    #    and it has to be 2-D. The output is also 2-D, and we just take the 1st row.
    # shape = [nb_tokens_to_mask]
    mask_types = tf.random.categorical(logits=tf.math.log([mask_type_probs]),
                                       num_samples=nb_tokens_to_mask)[0]

    # These are token ids after applying masking.
    # shape = [nb_tokens_to_mask]
    masked_input_ids = (
        _input_ids_mask * tf.cast(mask_types == 0, dtype=tf.int32) + \
        _input_ids_real * tf.cast(mask_types == 1, dtype=tf.int32) + \
        _input_ids_rand * tf.cast(mask_types == 2, dtype=tf.int32)
    )

    # Put the masked token ids into a 2-D tensor (initially zeros) of shape [batch_size, seq_len].
    # remark: `tf.where(pred_mask)` is of shape [nb_tokens_to_mask, 2].
    token_ids_to_updates = tf.scatter_nd(indices=tf.where(pred_mask),
                                         updates=masked_input_ids,
                                         shape=[batch_size, seq_len])

    # At the places where we don't mask, just keep the original token ids.
    # shape = [batch_size, seq_len]
    token_ids_to_keep = input_ids * tf.cast(~pred_mask, tf.int32)

    # The final masked token ids used for training
    # shape = [batch_size, seq_len]
    masked_input_ids = token_ids_to_updates + token_ids_to_keep

    # At the places where we don't predict, change the labels to -100
    # shape = [batch_size, seq_len]
    mlm_labels = input_ids * tf.cast(
        pred_mask, dtype=tf.int32) + -100 * tf.cast(~pred_mask, tf.int32)

    masked_lm_batch = {
        'input_ids': masked_input_ids,
        'attention_mask': attention_mask
    }
    if 'token_type_ids' in inputs:
      masked_lm_batch['token_type_ids'] = inputs['token_type_ids']

    # The total number of tokens
    nb_tokens = tf.reduce_sum(tf.cast(input_ids > -1, dtype=tf.int32))

    # Used for visualization
    # 0: not masked, 1: masked, 2: keep it as it is, 3: replaced by a random token, 4: padding - (not masked)
    # shape = [batch_size, seq_len]
    _mask_types = tf.scatter_nd(tf.where(pred_mask),
                                updates=mask_types + 1,
                                shape=[batch_size, seq_len])
    _mask_types = tf.cast(_mask_types, dtype=tf.int32)
    _mask_types += 4 * tf.cast(input_ids == pad_token_id, tf.int32)

    result = {
        'inputs':
            masked_lm_batch,
        'mlm_labels':
            mlm_labels,
        'mask_types':
            _mask_types,
        'original_input_ids':
            input_ids,
        'nb_tokens':
            nb_tokens * tf.constant(1, shape=[batch_size]),
        'nb_non_padding_tokens':
            nb_non_padding_tokens * tf.constant(1, shape=[batch_size]),
        'nb_tokens_considered':
            nb_tokens_considered * tf.constant(1, shape=[batch_size]),
        'nb_tokens_masked':
            nb_tokens_to_mask * tf.constant(1, shape=[batch_size])
    }

    return result

  return prepare_masked_lm_batch

class MyTokenizer():
  def __init__(self, vocab_size):
    self.vocab_size = vocab_size
    # # 0 pad， 1 unk 2 mask 101 cls 102 sep 
    self.pad_token_id = 0
    self.mask_token_id = 2
    # self.all_special_ids = [0, 1, 2, 101, 102]
    self.all_special_ids = list(range(200))

class Model(mt.Model):

  def __init__(self, 
               transformer=None, 
               bert=None,
               dense=None,
               tokenizer=None,
               custom_model=False,
               embedding_path=None, 
               vocab_size=None, 
               hidden_size=None, 
               num_attention_heads=None, 
               num_hidden_layers=None,
               default_model='bert-base-chinese',
               l2_norm=True, 
               token_counts=None,
               count_tokens=False,
               mlm_mask_prob=0.15,
               mask_type_probs=(0.8, 0.1, 0.1),
               predict_special_tokens=False,
               **kwargs):
    super().__init__(**kwargs)

    if bert is None:
      self.custom_model = custom_model
      ic(self.custom_model)
      if not custom_model:
        try:
          self.bert = TFAutoModelForMaskedLM.from_pretrained(transformer,
                                                            from_pt=False)
        except Exception:
          self.bert = TFAutoModelForMaskedLM.from_pretrained(transformer,
                                                            from_pt=True)
        tokenizer = Tokenizer.from_pretrained(transformer)
        config = AutoConfig.from_pretrained(transformer)
      else:
        self.custom_model = True

        if embedding_path and os.path.exists(embedding_path):
          embedding = np.load(embedding_path)
          if l2_norm:
            embedding = gezi.normalize(embedding)
          vocab_size = vocab_size or len(embedding)
          if vocab_size < len(embedding):
            embedding = embedding[:vocab_size]
          if not hidden_size:
            hidden_size = embedding.shape[1]
          ic(embedding)
          ic(vocab_size, hidden_size)
        else:
          assert vocab_size
        config = AutoConfig.from_pretrained(transformer)
        hidden_size = hidden_size or config.hidden_size
        num_attention_heads = num_attention_heads or config.num_attention_heads
        num_hidden_layers = num_hidden_layers or config.num_hidden_layers

        config.update(
        {
          "vocab_size": vocab_size,
          "hidden_size": hidden_size,
          "num_attention_heads": num_attention_heads,
          "num_hidden_layers": num_hidden_layers,
        })

        ic(config.vocab_size, config.hidden_size, config.num_hidden_layers)
        self.bert = TFAutoModelForMaskedLM.from_config(config)

        if embedding_path and os.path.exists(embedding_path):
          initializer = tf.keras.initializers.constant(embedding)
        else:
          initializer = 'uniform'
        with tf.name_scope("word_embeddings"):  
          word_embeddings = self.add_weight(
              "weight",
              shape=[config.vocab_size, config.hidden_size],
              initializer=initializer,
          )
        self.bert.bert.set_input_embeddings(word_embeddings)
        
        tokenizer = MyTokenizer(vocab_size)
        # tokenizer = Tokenizer.from_pretrained(transformer)
        # tokenizer.vocab_size = vocab_size
        # # 0 pad， 1 unk 2 mask 101 cls 102 sep 
        # tokenizer.pad_token_id = 0
        # tokenizer.mask_token_id = 2
        # tokenizer.all_special_ids = [0, 1, 2, 101, 102]

      # max_len = 64
      self.dense = tf.keras.layers.Dense(config.hidden_size)
    else:
      self.bert = bert
      self.dense = dense
      if not tokenizer:
        assert vocab_size
        tokenizer = MyTokenizer(vocab_size)

    self.prepare_masked_lm_batch = get_masked_lm_fn(tokenizer, mlm_mask_prob, 
                                    mask_type_probs, token_counts=token_counts, 
                                    predict_special_tokens=predict_special_tokens)

    
  def call(self, batch, embs=None, attention_mask=None, token_type_ids=None, training=None):
    batch_ = batch
    if not 'inputs' in batch:
      batch = self.prepare_masked_lm_batch(batch)
    inputs, mlm_labels, nb_tokens_masked = batch['inputs'], batch[
        'mlm_labels'], batch['nb_tokens_masked']
    # sequence outputs
    # shape = [batch_size, seq_len, vocab_size]
    if embs is None:
      if 'embs' in batch_:
        embs = batch_['embs']
        attention_mask = None
        if 'emb_attention_mask' in batch_:
          attention_mask = batch_['emb_attention_mask']
        token_type_ids = None
        if 'emb_token_type_ids' in batch_:
          token_type_ids = batch_['emb_token_type_ids']

    # tf.print(embs)
    if embs is None:
      logits = self.bert(inputs, training=True)[0]
    else:
      embs = self.dense(embs)
      embs0 = tf.gather(self.bert.bert.embeddings.weight, inputs['input_ids'])
      # if attention_mask is None:
      attention_mask = tf.fill([mt.get_shape(embs, 0), 
                                mt.get_shape(embs, 1)],
                                1)
      attention_mask = tf.concat([inputs['attention_mask'], tf.cast(attention_mask, inputs['attention_mask'].dtype)], -1)
      token_type_ids0 = inputs['token_type_ids'] if 'token_type_ids' in inputs else tf.zeros_like(inputs['input_ids'])
      if token_type_ids is None:
        val = tf.reduce_max(token_type_ids0) + 1
        token_type_ids = tf.cast(tf.fill([mt.get_shape(embs, 0), 
                                          mt.get_shape(embs, 1)],
                              val), inputs['input_ids'].dtype)
      token_type_ids = tf.concat([token_type_ids0, token_type_ids], -1)
      embs = tf.concat([embs0, embs], 1)
      len_ = inputs['input_ids'].shape[1]
      logits = self.bert(None, attention_mask, token_type_ids, inputs_embeds=embs)[0]
      # tf.print(logits, logits.shape)
      logits = logits[:,:len_]
      # tf.print(logits.shape)

    # get the places where the tokens should be predicted (masked / replaced / )
    # shape = [batch_size, seq_len]
    mlm_mask = (mlm_labels > -1)

    # shape = [nb_masked_tokens]
    labels_at_masked_tokens = tf.boolean_mask(mlm_labels, mlm_mask)

    # shape = [nb_masked_tokens, vocab_size]
    logits_at_masked_tokens = tf.boolean_mask(logits, mlm_mask)

    # tf.print(mlm_mask)
    ## TensorShape([1024, 32, 21128]) TensorShape([1024, 32])
    # tf.print(logits.shape, mlm_labels.shape)

    self.logits = logits_at_masked_tokens
    self.labels = labels_at_masked_tokens

    #  [-0.702047884 0.498109847 -0.355674803 ... 0.00116665289 2.47561193 -0.224801958]
    #  [0.546316504 1.32756495 0.529810786 ... 0.814874589 3.12588358 2.33146524]] [1925 4638 6817 ... 6821 2769 8024]
    # TensorShape([4176, 21128]) TensorShape([4176])
    # 2021-10-05 14:13:20 0:01:14 Model: "model"
    # tf.print(self.logits, self.labels)
    # tf.print(self.logits.shape, self.labels.shape)
    self.nb_tokens_masked = nb_tokens_masked

    return logits

  def get_loss_fn(self):
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.SUM)

    def loss_fn(y_true, y_pred):
      loss = loss_obj(tf.cast(self.labels, tf.float32), tf.cast(self.logits, tf.float32))
      loss = loss / tf.cast(self.nb_tokens_masked[0], dtype=tf.float32)
      return loss

    return loss_fn
