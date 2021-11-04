#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   optimizers.py
#        \author   chenghuige  
#          \date   2018-10-29 07:06:55.090940
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
FLAGS = flags.FLAGS

import tensorflow as tf 
import sys 
import os

# http://nlp.seas.harvard.edu/2018/04/03/attention.html
class OptWrapper:
    def __init__(self, optimizer, lr=0.):
        self._step = 0
        self._rate = 0.
        self.start_lr = lr
        self.optimizer = optimizer
        self.param_groups = self.optimizer.param_groups
        if self.start_lr:
            for p in self.optimizer.param_groups:
                p['ratio'] = p['lr'] / self.start_lr

    def set_step(self, step):
        self._step = step

    def step(self):
        "Update parameters and rate"
        self._step += 1

        rate = self.rate()

        for p in self.optimizer.param_groups:
            #p['lr'] = rate 
            if 'ratio' in p:
              p['lr'] = rate * p['ratio']

        self._rate = rate
        self.optimizer.step()

    def zero_grad(self):
      self.optimizer.zero_grad()

    def state_dict(self):
      return self.optimizer.state_dict()

    def load_state_dict(self, x):
        return self.optimizer.load_state_dict(x)

class NoamOpt(OptWrapper):
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        super(NoamOpt, self).__init__(optimizer)
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        
    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))  

def lr_poly(base_lr, iter, max_iter, end_learning_rate, power):
    return (base_lr - end_learning_rate) * ((1 - float(iter) / max_iter) ** (power)) + end_learning_rate

class BertOpt(OptWrapper):
    "Optim wrapper that implements learning rate."
    def __init__(self, lr, min_lr, num_train_steps, warmup, optimizer, power=1.):
        super(BertOpt, self).__init__(optimizer, lr)
        self.warmup = warmup
        self.lr = lr
        self.ori_min_lr = min_lr
        self.min_lr = min_lr
        self.num_train_steps = num_train_steps
        self.power = power
        #print('---------param_groups', self.optimizer.param_groups)
        
    def rate(self, step=None):
        #print('-------------here')
        "Implement `lrate` above"
        if step is None:
            step = self._step
            
        warmup_percent_done = step / self.warmup
        warmup_learning_rate = self.lr * warmup_percent_done 

        is_warmup = step < self.warmup
        learning_rate = lr_poly(self.lr, step, self.num_train_steps, self.min_lr, self.power)
        learning_rate = ((1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)
        #print('-----------------', is_warmup, warmup_percent_done, warmup_learning_rate, warmup_learning_rate)
        return learning_rate

    def update(self, num_train_steps, num_warmup_steps):
        self.num_train_steps = num_train_steps
        self.num_warmup_steps = num_warmup_steps

class MultipleOpt(object):
    def __init__(self, *optimizers):
        self.optimizers = optimizers

    def set_step(self, step):
        for op in self.optimizers:
            op._step = step

    def rate(self, step=None):
        return self.optimizers[0].rate(step)

    def rates(self, step=None):
        return [op.rate(step) for op in self.optimizers]

    @property
    def param_groups(self):
        param_groups = []
        for optimizer in self.optimizers:
            param_groups.extend(optimizer.param_groups)
        return param_groups

    def zero_grad(self):
        """ ? """
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        """ ? """
        for op in self.optimizers:
            op.step()

    @property
    def state(self):
        """ ? """
        return {k: v for op in self.optimizers for k, v in op.state.items()}

    def state_dict(self):
        """ ? """
        return [op.state_dict() for op in self.optimizers]

    def load_state_dict(self, state_dicts):
        assert len(state_dicts) == len(self.optimizers)
        for i in range(len(self.optimizers)):
            self.optimizers[i].load_state_dict(state_dicts[i])

if __name__ == '__main__':
  import matplotlib.pyplot as plt 
  import numpy as np

  steps=2326
  for i in range(steps):
    lr = lr_poly(0.1, i, steps, 1e-6, 1.)
    print(i, lr)

  opts = [NoamOpt(512, 1, 4000, None), 
          NoamOpt(512, 1, 8000, None),
          NoamOpt(256, 1, 4000, None),
          NoamOpt(200, 2, 4000, None),
          NoamOpt(256, 2, 4000, None),
          NoamOpt(300, 2, 4000, None),
          NoamOpt(128, 2, 4000, None)]
  plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
  plt.legend(["512:4000", "512:8000", "256:4000", "200:2:4000",  "256:2:4000", "300:2:4000", "128:2:4000"])

  for i in range(1, 40000, 1000):
      print(i, NoamOpt(200, 2, 2000, None).rate(i))

  plt.savefig('/home/gezi/tmp/lr.png')           
