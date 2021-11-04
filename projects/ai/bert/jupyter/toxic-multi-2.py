
# coding: utf-8

# In[1]:


import sys
import os  
import traceback

RECORDS_PATH = '../input/tfrecords'

if os.path.exists('/kaggle'):
  sys.path.append('/kaggle/input/gezi-melt/utils')
  sys.path.append('/kaggle/input/official')
  from kaggle_datasets import KaggleDatasets
  try:
    GCS_PATH = KaggleDatasets().get_gcs_path('jigsaw-multilingual-toxic-comment-classification')
    RECORDS_PATH = KaggleDatasets().get_gcs_path('toxic-multi-tfrecords') + '/tfrecords'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
  except Exception:
    print(traceback.format_exc())
    RECORDS_PATH = '../input/toxic-multi-tfrecords/tfrecords'
    pass


# In[2]:


import official
import gezi
import melt
import lele
import husky
import pandas as pd
import numpy as np
import tensorflow as tf
tf.__version__
from absl import app, flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', None, '')
flags.DEFINE_bool('multi_head', False, '')
flags.DEFINE_string('pretrained', '../input/tf-xlm-roberta-large', '')
flags.DEFINE_integer('max_len', 192, 'xlm 192 bert 128')
flags.DEFINE_bool('freeze_pretrained', False, '')


# In[3]:


# flags
argv=['']
FLAGS(argv)
mark='xlm'
FLAGS.train_input=f'{RECORDS_PATH}/{mark}/jigsaw-toxic-comment-train'
FLAGS.valid_input=f'{RECORDS_PATH}/{mark}/validation'
FLAGS.test_input=f'{RECORDS_PATH}/{mark}/test'
FLAGS.valid_interval_steps=100 
FLAGS.verbose=1 
FLAGS.num_epochs=1
FLAGS.keras=1 
FLAGS.buffer_size=2048
FLAGS.learning_rate=1e-5 
FLAGS.min_learning_rate=0.
# FLAGS.opt_epsilon=1e-8 
# FLAGS.optimizer='bert-adamw'
FLAGS.optimizer='adam'
FLAGS.metrics=[] 
FLAGS.test_names=['id', 'toxic']
FLAGS.valid_interval_epochs=0.1
FLAGS.test_interval_epochs=1.
FLAGS.num_gpus=4
FLAGS.cache=0
FLAGS.model_dir='../working/exps/v1/base'
FLAGS.multi_head=0
FLAGS.batch_parse=1
FLAGS.save_model=0
# FLAGS.pretrained = '../input/tf-xlm-roberta-large/'
FLAGS.pretrained = '../input/tf-xlm-roberta-base/'
FLAGS.batch_size=16 if 'large' in FLAGS.pretrained else 32
# FLAGS.batch_size=16
FLAGS.debug=0

toxic_types = ['severe_toxic', 'obscene', 'identity_hate', 'threat', 'insult']


# In[4]:


# evaluate
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.utils import shuffle

import gezi
logging = gezi.logging

langs = ['es', 'it', 'tr']

def evaluate(y_true, y_pred, x):
  try:
    y_true = y_true[:,0]
    y_pred = y_pred[:,0]
  except Exception:
    pass
  if y_pred.max() > 1. or y_pred.min() < 0:
    y_pred = gezi.sigmoid(y_pred)
  result = {}
  loss = log_loss(y_true, y_pred)
  result['loss'] = loss
  
  auc = roc_auc_score(y_true, y_pred)
  result['auc/all'] = auc
    
  if 'lang' in x:
    x['y_true'] = y_true
    x['pred'] = y_pred
    x['lang'] = gezi.decode(x['lang'])

    df = pd.DataFrame(x) 
    df = shuffle(df)
    logging.info('\n', df)

    for lang in langs:
      df_ = df[df.lang==lang]
      auc = roc_auc_score(df_.y_true, df_.pred)
      result[f'auc/{lang}'] = auc

  return result


# In[5]:


# dataset
import tensorflow as tf
from tensorflow.keras import backend as K
import melt

class Dataset(melt.Dataset):
  def __init__(self, subset='valid', **kwargs):
    super(Dataset, self).__init__(subset, **kwargs)

  def parse(self, example):
    MAX_LEN = FLAGS.max_len
    features_dict = {
      'input_word_ids': tf.io.FixedLenFeature([MAX_LEN], tf.int64),
      'toxic': tf.io.FixedLenFeature([], tf.float32),
      'id': tf.io.FixedLenFeature([], tf.string),
      # 'comment_text': tf.io.FixedLenFeature([], tf.string), # TODO
    }

    def _adds(names, dtype=None, length=None):
      dtype_ = dtype
      for name in names:
        if name in self.example:
          dtype = dtype_ or self.example[name][0].dtype 
          if length:
            features_dict[name] = tf.io.FixedLenFeature([length], dtype)
          else:
            features_dict[name] = tf.io.FixedLenFeature([], dtype)

    _adds(['lang'], tf.string)

    _adds(['input_mask', 'all_segment_id'], tf.int64, MAX_LEN)
    
    _adds(toxic_types)

    features = self.parse_(serialized=example, features=features_dict)

    def _casts(names, dtype=tf.int32):
      for name in names:
        if name in features:
          features[name] = tf.cast(features[name], dtype)

    _casts(['input_word_ids', 'input_mask', 'all_segment_id'])
    
    x = features
    y = features['toxic']
#     y = tf.cast(features['toxic'] > 0.5, tf.float32)
    keys = ['toxic', *toxic_types]
    for key in keys:
      if key not in features:
        features[key] = tf.zeros_like(features['toxic'])
        
    _casts(toxic_types, tf.float32)
        
    melt.append_dim(features, keys)

    if FLAGS.multi_head:
      y = tf.concat([features[key] for key in keys], 1)

    return x, y
 


# In[6]:


# loss
import tensorflow as tf

def calc_loss(y_true, y_pred):
  pass

def focal_loss(gamma=1.5, alpha=.2):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

def get_loss_fn():
#   return tf.compat.v1.losses.sigmoid_cross_entropy
  return tf.keras.losses.BinaryCrossentropy()
#   return focal_loss()


# In[7]:


# model
import tensorflow as tf
import tensorflow_hub as hub

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Input

import melt
import gezi 
logging = gezi.logging

# class Model(keras.Model):
#   def __init__(self):
#     super(Model, self).__init__() 

#     self.bert_layer = bert_layer
#     dims = [32]
#     self.mlp = melt.layers.MLP(dims)
#     odim = len(toxic_types) + 1 if FLAGS.multi_head else 1
#     self.dense = keras.layers.Dense(odim, activation='sigmoid')

#   def call(self, input):
#     input_word_ids = input['input_word_ids']
#     input_mask = input['input_mask']
#     segment_ids = input['all_segment_id']
  
#     x, _ = self.bert_layer([input_word_ids, input_mask, segment_ids])
#     x = self.mlp(x)
#     x = self.dense(x)
#     return x


# In[8]:


import transformers
from transformers import TFAutoModel

def xlm_model():
  pretrained = FLAGS.pretrained or XLM_PATH
  with gezi.Timer(f'load xlm_model from {pretrained}', True, logging.info):
    transformer = TFAutoModel.from_pretrained(pretrained)
  if FLAGS.freeze_pretrained:
    transformer.trainable = False
  input_word_ids = Input(shape=(FLAGS.max_len,), dtype=tf.int32, name="input_word_ids")
  sequence_output = transformer(input_word_ids)[0]
  cls_token = sequence_output[:, 0, :]
  odim = len(toxic_types) + 1 if FLAGS.multi_head else 1
  out = keras.layers.Dense(odim, activation='sigmoid')(cls_token)

  model = keras.Model(inputs=input_word_ids, outputs=out)

  return model


# In[9]:


Model = xlm_model


# In[10]:


# train

import os
import melt

fit = melt.fit
melt.init()
loss_fn = get_loss_fn()


# In[11]:

strategy = melt.distributed.get_strategy()
with strategy.scope():
  model = Model()
try:
  model.summary()
except Exception:
  pass
# In[12]:


# model.load_weights('../input/toxic-multi/xlm.toxic.h5')


# In[13]:


def run(model=model):
  fit(model,  
      loss_fn,
      Dataset,
      eval_fn=evaluate,
      eval_keys=['lang'],
      )   


# In[ ]:


FLAGS.train_input=f'{RECORDS_PATH}/xlm-sample1/jigsaw-unintended-bias-train'
FLAGS.learning_rate=1e-5
# FLAGS.opt_epsilon=1e-8
FLAGS.num_epochs=1
run()


# In[ ]:


FLAGS.train_input=f'{RECORDS_PATH}/{mark}/jigsaw-toxic-comment-train'
FLAGS.learning_rate=1e-5
# FLAGS.opt_epsilon=1e-8
FLAGS.num_epochs=1  
run()
# # model.save('./xlm.toxic.3e5.h5')


# In[ ]:


# cv
FLAGS.train_input=FLAGS.valid_input
FLAGS.learning_rate=1e-5
valid_input = FLAGS.valid_input
FLAGS.num_folds = 5
FLAGS.vie=1.
run()
FLAGS.num_folds = None
FLAGS.vie=0.1
FLAGS.valid_input = FLAGS.train_input


# In[ ]:


# model.save_weights('./xlm.toxic-uint1.h5')


# In[ ]:


FLAGS.train_input=FLAGS.valid_input
FLAGS.learning_rate=1e-5
# FLAGS.opt_epsilon=1e-8
FLAGS.num_epochs=1
run()


# In[ ]:


# model.save_weights('./xlm.final.h5')


# In[ ]:


# # with strategy.scope():
# #   model = Model()
# FLAGS.train_input=f'{RECORDS_GCS_PATH}/{mark}/jigsaw-toxic-comment-train,{RECORDS_GCS_PATH}/xlm-sample2/jigsaw-unintended-bias-train'
# # FLAGS.train_input=f'{RECORDS_GCS_PATH}/xlm/jigsaw-unintended-bias-train'
# FLAGS.learning_rate=3e-5
# FLAGS.opt_epsilon=1e-8
# FLAGS.num_epochs=1  
# FLAGS.valid_interval_epochs=0.1
# run()


# In[ ]:


# FLAGS.train_input=FLAGS.valid_input
# FLAGS.learning_rate=3e-5
# FLAGS.opt_epsilon=1e-8
# FLAGS.num_epochs=1
# FLAGS.valid_interval_epochs=0.2
# FLAGS.optimizer='bert-adamw'
# run()


# In[ ]:


d = pd.read_csv('../working/exps/v1/base/submission.csv')
d


# In[ ]:


d.to_csv('submission.csv', index=False)
d.head()

