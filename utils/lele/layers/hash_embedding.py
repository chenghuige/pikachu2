#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   hash_embedding.py
#        \author   chenghuige  
#          \date   2019-12-31 22:48:55.756940
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals, division

import sys 
import os
import numpy as np
import traceback

from absl import flags
FLAGS = flags.FLAGS

import torch
import torch.nn as nn
from torch.nn.init import normal

import gezi
logging = gezi.logging
import lele


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

    def __init__(self, bins, mask_zero=False, moduler=None):
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

class HashEmbedding(nn.Module):
    r"""Type of embedding which uses multiple hashes to approximate an Embedding layer using less parameters.

    This module is a new Embedding module that compresses the number of parameters. They are a
    generalization of vanilla Embeddings and the `hashing trick`. For more details, check Svenstrup,
    Dan Tito, Jonas Hansen, and Ole Winther. "Hash embeddings for efficient word
    representations." Advances in Neural Information Processing Systems. 2017.

    For each elements (usually word indices) in the input (mini_batch, sequence_length) the default
    computations are:

    .. math::

            \begin{array}{ll}
            H_i = E_{D_2^i(D_1(w)))} \ \forall i=1...k\\
            c_w = (H_1(w), ..., H_k(w))^T\\
            p_w = P_{D_1(w)}\\
            \hat{e}_w = p_w \cdot c_w\\
            e_w = \mathrm{concatenate}(\hat{e}_w,p_w)\\
            \end{array}

    where :math:`w:[0,T]` is the element of the input (word index), :math:`D_1:[0,T)\to [0,K)`
    is the token to ID hash/dictionnary, :math:`D_2:[0,K)\to[0,B)` is the ID to Bucket hash,
    :math:`E:\mathbb R^{B*d}` is the shared pool of embeddings, :math:`c_w:\mathbb R^{k*d}` contains all
    the vector embeddings to which :math:`w` maps, :math:`e_w:\mathbb R^{d+k}` is the outputed word
    embedding for :math:`w`.

    Args:
        num_embeddings (int): the number of different embeddings. K in the paper.
            Higher increases possible vocabulary size.
        embedding_dim (int): the size of each embedding vector in the shared pool. d in the paper.
            Higher improves downstream task for fixed vocabulary size.
        num_buckets (int,optional): the size of the shared pool of embeddings. B in the paper.
            Higher improves approximation quality. Typically num_buckets * 10 < num_embeddings.
        num_hashes (int,optional): the number of different hash functions. k in the paper.
            Higher improves approximation quality. Typically in [1,3].
        train_sharedEmbed (bool,optional): whether to train the shared pool of embeddings E.
        train_weight (bool,optional): whether to train the importance parameters / weight P.
        append_weight (bool,optional): whether to append the importance parameters / weight pw.
        aggregation_combiner ({"sum","median","concatenate"},optional): how to aggregate the (weighted) component
            vectors of the different hashes. Sum should be the same as mean (because learnable parameters,
            can learn to divide by n)
        mask_zero (bool, optional): whether the 0 input is a special "padding" value to mask out.
        seed (int, optional): sets the seed for generating random numbers.
        oldAlgorithm (bool, optional): whether to use the algorithm in the paper rather than the improved version.
            I do not recommend to set to true besides for comparaison.

    Attributes:
        shared_embeddings (nn.Embedding): the shared pool of embeddings of shape (num_buckets, embedding_dim).
            E in the paper.
        importance_weights (nn.Embedding): the importance parameters / weight of shape
            (num_embeddings, num_hashes). P in the paper.
        output_dim (int): effective outputed number of embeddings.

    Shape:
        - Input: LongTensor `(N, W)`, N = mini-batch, W = number of indices to extract per mini-batch
        - Output: `(N, W, output_dim)`, output_dim is the effective embedding dim.

    Examples::
        >>> # an HashEmbedding module containing approximating nn.Embedding(10, 5) with less param
        >>> embedding = HashEmbedding(10,5,append_weight=False)
        >>> # a batch of 2 samples of 4 indices each
        >>> input = Variable(torch.LongTensor([[1,2,4,5],[4,3,2,9]]))
        >>> embedding(input)

        Variable containing:
        (0 ,.,.) =
        1.00000e-04 *
           0.3988  0.5234 -0.6148  0.3000 -1.5525
           0.1259  0.4142 -0.8613  0.3018 -1.3547
           0.1367  0.2638 -0.2993  0.9541 -1.7194
          -0.4672 -0.7971 -0.2009  0.7829 -0.9448

        (1 ,.,.) =
        1.00000e-04 *
           0.1367  0.2638 -0.2993  0.9541 -1.7194
          -0.0878 -0.1680  0.3896  0.5288 -0.2060
           0.1259  0.4142 -0.8613  0.3018 -1.3547
          -0.3098  0.0357 -0.7532 -0.1216 -0.0366
        [torch.FloatTensor of size 2x4x5]

        >>> # example with mask_zero which corresponds to padding_idx=0
        >>> embedding = HashEmbedding(10,5,append_weight=False, mask_zero=True)
        >>> input = Variable(torch.LongTensor([[0,2,0,5]]))
        >>> embedding(input)

        Variable containing:
        (0 ,.,.) =
        1.00000e-04 *
           0.0000  0.0000  0.0000  0.0000  0.0000
          -1.4941 -1.3775 -0.5797  1.3187 -0.0555
           0.0000  0.0000  0.0000  0.0000  0.0000
          -0.7717 -0.5569 -0.1397  1.1101 -0.0939
        [torch.FloatTensor of size 1x4x5]
    """

    def __init__(self, num_embeddings, embedding_dim, num_buckets=None, num_hashes=2, train_sharedEmbed=True,
                 train_weight=True, append_weight=True, combiner='sum', padding_idx=None, sparse=False, 
                 seed=None, oldAlgorithm=False):
        super(HashEmbedding, self).__init__()

        agrregation_combiner = combiner
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_hashes = num_hashes
        defaultNBuckets = (num_embeddings * self.num_hashes) // (self.embedding_dim)
        self.num_buckets = num_buckets - 1 if num_buckets is not None else defaultNBuckets
        self.train_sharedEmbed = train_sharedEmbed
        self.train_weight = train_weight
        self.append_weight = append_weight
        self.padding_idx = padding_idx
        self.seed = seed
        # THERE IS NO ADVANTAGE OF SETTING THE FOLLOWING TO TRUE, I JUST HAVE TO COMPARE WITH THE ALGORITHM IN THE PAPER
        self.oldAlgorithm = oldAlgorithm

        self.importance_weights = nn.Embedding(self.num_embeddings,
                                               self.num_hashes,
                                               sparse=sparse)
        self.shared_embeddings = nn.Embedding(self.num_buckets + 1,
                                              self.embedding_dim,
                                              padding_idx=self.padding_idx,
                                              sparse=sparse)

        mask_zero = padding_idx is not None
        hashFamily = HashFamily(self.num_buckets, mask_zero=mask_zero)
        self.hashes = hashFamily.draw_hashes(self.num_hashes)

        if aggregation_combiner == 'sum':
            self.aggregate = lambda x: torch.sum(x, dim=-1)
        elif aggregation_combiner == 'mul':
            self.aggregate = lambda x: torch.prod(x, dim=-1)
        elif aggregation_combiner == 'concatenate':
            # little bit quicker than permute/contiguous/view
            self.aggregate = lambda x: torch.cat([x[:, :, :, i] for i in range(self.num_hashes)], dim=-1)
        elif aggregation_combiner == 'median':
            print('median')
            self.aggregate = lambda x: torch.median(x, dim=-1)[0]
        else:
            raise ValueError('unknown aggregation function {}'.format(aggregation_combiner))

        self.output_dim = self.embedding_dim
        if aggregation_combiner == "concatenate":
            self.output_dim *= self.num_hashes
        if self.append_weight:
            self.output_dim += self.num_hashes

        self.reset_parameters()

    def reset_parameters(self,
                         init_shared=lambda x: normal(x, std=0.1),
                         init_importance=lambda x: normal(x, std=0.0005)):
        """Resets the trainable parameters."""
        def set_constant_row(parameters, iRow=0, value=0):
            """Return `parameters` with row `iRow` as s constant `value`."""
            data = parameters.data
            data[iRow, :] = value
            return torch.nn.Parameter(data, requires_grad=parameters.requires_grad)

        np.random.seed(self.seed)
        if self.seed is not None:
            torch.manual_seed(self.seed)

        self.shared_embeddings.weight = init_shared(self.shared_embeddings.weight)
        self.importance_weights.weight = init_importance(self.importance_weights.weight)

        if self.padding_idx is not None:
            # Unfortunately has to set weight to 0 even when paddingIdx = 0
            self.shared_embeddings.weight = set_constant_row(self.shared_embeddings.weight)
            self.importance_weights.weight = set_constant_row(self.importance_weights.weight)

        self.shared_embeddings.weight.requires_grad = self.train_sharedEmbed
        self.importance_weights.weight.requires_grad = self.train_weight

    def forward(self, input):
        idx_importance_weights = input % self.num_embeddings
        # THERE IS NO ADVANTAGE OF USING THE FOLLWOING LINE, I JUST HAVE TO COMPARE WITH THE ALGORITHM IN THE PAPER
        input = idx_importance_weights if self.oldAlgorithm else input
        idx_shared_embeddings = torch.stack([h(input).masked_fill_(input == 0, 0) for h in self.hashes], dim=-1)

        shared_embedding = torch.stack([self.shared_embeddings(idx_shared_embeddings[:, :, iHash])
                                        for iHash in range(self.num_hashes)], dim=-1)
        importance_weight = self.importance_weights(idx_importance_weights)
        importance_weight = importance_weight.unsqueeze(-2)
        word_embedding = self.aggregate(importance_weight * shared_embedding)
        if self.append_weight:
            # concateates the vector with the weights
            word_embedding = torch.cat([word_embedding, importance_weight.squeeze(-2)], dim=-1)
        return word_embedding

# class SimpleEmbedding(nn.Embedding):
#     def __init__(self, num_embeddings, embedding_dim, num_buckets=None, combiner=None, mode=None, **kwargs):
#         super(SimpleEmbedding, self).__init__(num_buckets, embedding_dim, **kwargs)
#         self.num_embeddings = num_buckets
#         self.num_buckets = num_buckets

#     def forward(self, input):
#         return super(SimpleEmbedding, self).forward(input % self.num_buckets)

class SimpleEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_buckets=None, combiner=None, 
                 mode=None, split=False, large_emb=False, **kwargs):
        super(SimpleEmbedding, self).__init__()
        self.num_embeddings = num_buckets
        self.num_buckets = num_buckets
        self.large_emb = large_emb
        if not large_emb:
          self.embedding = nn.Embedding(self.num_buckets + 1, embedding_dim, **kwargs)

    def forward(self, x):
        mask = x.ne(0).type(x.dtype)
        x = x % self.num_buckets
        x = (x + 1) * mask
        return self.embedding(x)

class QREmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_buckets=None, combiner='mul', 
                 mode=None, split=False, large_emb=False, **kwargs):
        super(QREmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_buckets = num_buckets
        self.num_quotients = -(-num_embeddings // num_buckets)
        self.large_emb = large_emb
        self.kwargs = kwargs
        if not large_emb:
          self.embedding = nn.Embedding(self.num_buckets + 1, embedding_dim, **kwargs)
        
        if self.num_embeddings != self.num_buckets:
          self.embedding2 = nn.Embedding(self.num_quotients + 1, embedding_dim, **kwargs)
        self.combiner = combiner

    def build(self):
      if self.large_emb:
        self.embedding = lele.layers.LargeEmbedding(self.num_buckets + 1, self.embedding_dim, **self.kwargs)

    def forward(self, x, combiner=None):
        mask = x.ne(0).type(x.dtype)
        x = x % self.num_embeddings
        if self.num_embeddings == self.num_buckets:
            return self.embedding(x)
        
        # TODO only use mask if self.padding_idx not None mask should be x.ne(self.padding_idx)
        # TODO why tf.cast to int32 will turn much slower *2 times..
        x_rem = (x % self.num_buckets + 1) * mask
        # NOTICE do not use cast like tf.cast(x/self.num_buckets, tf.int32)
        x_quo = (x // self.num_buckets + 1) * mask
        embs = self.embedding(x_rem)
        embs2 = self.embedding2(x_quo)
        combiner = combiner or self.combiner
        if combiner == 'mul':
            return embs * embs2
        elif combiner == 'sum':
            # -> [batch_size, len, out_dim]
            return embs + embs2
        elif combiner == 'concat':
            return torch.cat([embs, embs2], -1)
        else:
            raise ValueError(combiner)

class EmbeddingBags(nn.Module):
  def __init__(self, input_dim, output_dim, num_buckets, 
               padding_idx=None, sparse=False, 
               combiner=None, Embedding=None, split=False,
               return_list=False, mode='sum', **kwargs):
    super(EmbeddingBags, self).__init__(**kwargs)
    if Embedding is None:
      Embedding = QREmbedding
    self.Embedding = Embedding
    self.input_dim, self.output_dim = input_dim, output_dim
    self.num_buckets = num_buckets
    self.combiner = combiner
    self.kwargs = kwargs
    self.pooling = lele.layers.Pooling(mode) if mode else nn.Identity()
    self.mode = mode
    self.split = split
    self.return_list = return_list
    self.padding_idx = padding_idx
    self.sparse = sparse

    self.build()

  def build(self):
    # need to modify self.keys so copy 
    self.keys = gezi.get_global('embedding_keys').copy()
    try:
      if FLAGS.masked_fields:
        masked_keys = FLAGS.masked_fields.split(',')
        mask_mode = FLAGS.mask_mode.replace('_', '-').split('-')[-1]
        if 'regex' in FLAGS.mask_mode:
          if mask_mode == 'excl':
            def _is_ok(x):
              for key in masked_keys:
                if re.search(key, x):
                  return False
              return True
          else:
            def _is_ok(x):
              for key in masked_keys:
                if re.search(key, x):
                  return True
              return False
          self.keys = [x for x in self.keys if _is_ok(x)]
        else:  
          if mask_mode == 'excl':
            self.keys = [x for x in self.keys if x not in masked_keys]
          else:
            self.keys = masked_keys
        logging.debug('Final used onehot fields is:', ','.join(self.keys), 'count:', len(self.keys))
      if FLAGS.max_fields:
        np.random.shuffle(self.keys)
        self.keys = self.keys[:FLAGS.max_fields]
      gezi.set_global('embedding_keys', list(self.keys))
    except Exception:
      logging.warning(traceback.format_exc())
      pass

    num_buckets_ = int(self.num_buckets / len(self.keys))
    
    if not self.split:
      embedding = self.Embedding(self.input_dim, self.output_dim, self.num_buckets,
                                 padding_idx=self.padding_idx, sparse=self.sparse,
                                 mode=self.mode, combiner=self.combiner, 
                                 **self.kwargs)
      self.embedding = embedding
    else:
        key_to_idims = gezi.get_global('embedding_input_dims', {})
        key_to_odims = gezi.get_global('embedding_output_dims', {})
        self.embeddings = {}
        self.linears = {}
        embedding_infos = gezi.get_global('embedding_infos', {})
        for key in self.keys:
          embedding_info = embedding_infos.get(key, {})
          num_buckets = key_to_idims.get(key, num_buckets_)
          output_dim = key_to_odims.get(key, self.output_dim)
          Embedding = self.Embedding
          if 'type' in embedding_info:
            Embedding = getattr(lele.layers.hash_embedding, embedding_info['type'])
          num_buckets = embedding_info.get('input_dim', num_buckets)
          if num_buckets <= 20000:
            Embedding = SimpleEmbedding
          output_dim = embedding_info.get('output_dim', output_dim)
          mode = embedding_info.get('pooling', self.mode)
          if self.output_dim == 1:
            output_dim = 1
          input_dim = self.input_dim
          # input_dim = num_buckets * 10
          self.embeddings[key] = Embedding(input_dim, output_dim, num_buckets,
                                        padding_idx=self.padding_idx, sparse=self.sparse,
                                        mode=mode, combiner=self.combiner, 
                                        **self.kwargs)
          # logging.debug(key, Embedding, num_buckets, output_dim, pooling)
          if not hasattr(self.embeddings[key], 'pooling'):
            self.embeddings[key].pooling = None
          ## for to pooling in EmbeddingBags
          # self.embeddings[key].pooling = None
          if output_dim != self.output_dim:
            self.linears[key] = nn.Linear(output_dim, self.output_dim)
          else:
            self.linears[key] = nn.Identity()

        self.embeddings = nn.ModuleDict(self.embeddings)
        self.linears = nn.ModuleDict(self.linears)

#   def forward(self, x, value=None, key=None, pooling=True):
#     if key is not None:
#       return self.deal(key, x, value)
#     l = []
#     for i, key in enumerate(self.keys):
#       emb = self.deal(key, x, value, pooling=pooling)
#       l.append(emb)
#     if self.return_list:
#       return l
#     else:
#       return torch.stack(l, 1)

  def forward(self, x, value=None, key=None, pooling=True):
    if key is not None:
      return self.deal(key, x, value)
    if self.split:
      l = []
      for i, key in enumerate(self.keys):
        emb = self.deal(key, x, value, pooling=pooling)
        l.append(emb)
      if self.return_list:
        return l
      else:
        return torch.stack(l, 1)
    else:
      l = []
      inputs = []
      values = []
      input_lens = []
      real_lens = []
      for i, key in enumerate(self.keys):
        input_lens.append(x[key].shape[1])
        if value is not None:
          values.append(value[key])
        real_lens.append(None if self.mode == 'sum' else x[key].eq(0).float())
        inputs.append(x[key])
      input = torch.cat(inputs, 1)
      if value is not None:
        value = torch.cat(values, 1)
      embs = self.embedding(input)
      if value is not None:
        embs *= value.unsqueeze(-1)
      l = torch.split(embs, input_lens, 1)
      l = [self.pooling(x, len_) for x, len_ in zip(l, real_lens)]
        
      if self.return_list:
        return l
      else:
        return torch.stack(l, 1)

  def deal(self, key, x, value=None, pooling=True):
    if self.split:
      embedding = self.embeddings[key]
    else:
      embedding = self.embedding
    if pooling and hasattr(embedding, 'pooling') and embedding.pooling:
      # TODO support
      emb = embedding(x[key], value[key], pooling=True)
    else:
      embs = embedding(x[key])
      if value is not None and key in value:
        embs *= value[key].unsqueeze(-1)
        mask_ = None if self.mode == 'sum' else x.eq(0).float()
      else:
        mask_ = x.eq(0).float() 
      if pooling:
        emb = self.pooling(embs, mask_)
    if self.split:
      emb = self.linears[key](emb)
    return emb

  def get_embedding(key):
    if self.split:
      return self.embeddings[key]
    else:
      return self.embedding
