
import copy
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.keras.layers import Dot, Embedding, Flatten
import tensorflow_probability as tfp

from transformers import TFAutoModel, TFAutoModelForMaskedLM, TFAutoModelForSequenceClassification
from transformers import AutoConfig, BertConfig, TFBertModel
import global_objectives

import gezi
import melt as mt
from ..config import *
from .. import util

# model.py 用于之前模型迭代 可能有较多if else 很多分支不会走到
# model2.py 是当前迭代最优模型 并且去掉大部分未走到的实验分支(确定无效的)

class NeXtVLAD(tf.keras.layers.Layer):

  def __init__(self,
               feature_size,
               cluster_size,
               output_size=1024,
               expansion=2,
               groups=8,
               dropout=0.2):
    super().__init__()
    self.feature_size = feature_size
    self.cluster_size = cluster_size
    self.expansion = expansion
    self.groups = groups

    self.new_feature_size = expansion * feature_size // groups
    self.expand_dense = tf.keras.layers.Dense(self.expansion *
                                              self.feature_size)
    # for group attention
    self.attention_dense = tf.keras.layers.Dense(self.groups,
                                                 activation=tf.nn.sigmoid)
    # self.activation_bn = tf.keras.layers.BatchNormalization()

    # for cluster weights
    self.cluster_dense1 = tf.keras.layers.Dense(self.groups * self.cluster_size,
                                                activation=None,
                                                use_bias=False)
    # self.cluster_dense2 = tf.keras.layers.Dense(self.cluster_size, activation=None, use_bias=False)
    self.dropout = tf.keras.layers.Dropout(rate=dropout, seed=1)
    self.fc = tf.keras.layers.Dense(output_size, activation=None)

  def build(self, input_shape):
    self.cluster_weights2 = self.add_weight(
        name="cluster_weights2",
        shape=(1, self.new_feature_size, self.cluster_size),
        initializer=tf.keras.initializers.glorot_normal,
        trainable=True)
    self.built = True

  def call(self, inputs, **kwargs):
    image_embeddings, mask = inputs
    _, num_segments, _ = image_embeddings.shape
    if mask is not None:  # in case num of images is less than num_segments
      images_mask = tf.sequence_mask(mask, maxlen=num_segments)
      images_mask = tf.cast(tf.expand_dims(images_mask, -1),
                            image_embeddings.dtype)
      image_embeddings = tf.multiply(image_embeddings, images_mask)
    inputs = self.expand_dense(image_embeddings)
    attention = self.attention_dense(inputs)

    attention = tf.reshape(attention, [-1, num_segments * self.groups, 1])
    reshaped_input = tf.reshape(inputs,
                                [-1, self.expansion * self.feature_size])

    activation = self.cluster_dense1(reshaped_input)
    # activation = self.activation_bn(activation)
    activation = tf.reshape(activation,
                            [-1, num_segments * self.groups, self.cluster_size])
    # shape: batch_size * (max_frame*groups) * cluster_size
    activation = tf.nn.softmax(activation, axis=-1)
    # shape: batch_size * (max_frame*groups) * cluster_size
    activation = tf.multiply(activation, attention)

    # shape: batch_size * 1 * cluster_size
    a_sum = tf.reduce_sum(activation, -2, keepdims=True)
    # shape: batch_size * new_feature_size * cluster_size
    a = tf.multiply(a_sum, self.cluster_weights2)
    # shape: batch_size * cluster_size * (max_frame*groups)
    activation = tf.transpose(activation, perm=[0, 2, 1])

    reshaped_input = tf.reshape(
        inputs, [-1, num_segments * self.groups, self.new_feature_size])

    # shape: batch_size * cluster_size * new_feature_size
    vlad = tf.matmul(activation, reshaped_input)
    # shape: batch_size * new_feature_size * cluster_size
    vlad = tf.transpose(vlad, perm=[0, 2, 1])
    vlad = tf.subtract(vlad, a)
    vlad = tf.nn.l2_normalize(vlad, 1)
    self.feats = tf.reshape(vlad, [-1, self.cluster_size, self.new_feature_size])
    vlad = tf.reshape(vlad, [-1, self.cluster_size * self.new_feature_size])

    vlad = self.dropout(vlad)
    vlad = self.fc(vlad)
    return vlad

class SENet(tf.keras.layers.Layer):

  def __init__(self, channels, ratio=8, **kwargs):
    super(SENet, self).__init__(**kwargs)
    self.fc = tf.keras.Sequential([
        tf.keras.layers.Dense(channels // ratio,
                              activation='relu',
                              kernel_initializer='he_normal',
                              use_bias=False),
        tf.keras.layers.Dense(channels,
                              activation='sigmoid',
                              kernel_initializer='he_normal',
                              use_bias=False)
    ])

  def call(self, inputs, **kwargs):
    se = self.fc(inputs)
    outputs = tf.math.multiply(inputs, se)
    return outputs

# TODO change to Fusion
class Fusion(tf.keras.layers.Layer):
  def __init__(self, hidden_size, se_ratio, **kwargs):
    super().__init__(**kwargs)
    ic(mt.activation(FLAGS.activation))

    if FLAGS.mdrop or FLAGS.mdrop_rate > 0:
      self.fusion = mt.layers.MultiDropout(dims=[*FLAGS.mlp_dims, hidden_size], 
                                           activation=mt.activation(FLAGS.activation), 
                                           drop_rate=FLAGS.mdrop_rate,
                                           num_experts=FLAGS.mdrop_experts,
                                           activate_last=FLAGS.activate_last)
    else:
      self.fusion = mt.layers.MLP([*FLAGS.mlp_dims, hidden_size], activation=mt.activation(FLAGS.activation), activate_last=FLAGS.activate_last)

    self.fusion_dropout = tf.keras.layers.Dropout(FLAGS.fusion_dropout)
    if FLAGS.use_se:
      self.enhance = SENet(channels=hidden_size, ratio=se_ratio)
    if FLAGS.sfu:
      self.sfu = mt.layers.SemanticFusionCombine()
    if FLAGS.gem_norm:
      self.gem_norm = mt.layers.GeM(8)
    if FLAGS.batch_norm:
      self.batch_norm = tf.keras.layers.BatchNormalization()
    if FLAGS.layer_norm:
      self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    if FLAGS.pooling:
      self.pooling = mt.layers.Pooling(FLAGS.pooling)
      self.denses = [
        tf.keras.layers.Dense(FLAGS.vlad_hidden_size),
        tf.keras.layers.Dense(FLAGS.vlad_hidden_size),
        tf.keras.layers.Dense(FLAGS.vlad_hidden_size),
        tf.keras.layers.Dense(FLAGS.vlad_hidden_size)
      ] 

  def call(self, inputs, **kwargs):
    if not FLAGS.sfu:
      embeddings = tf.concat(inputs, axis=1)
    else:
      embeddings = self.sfu(inputs[0], inputs[1])

    if FLAGS.pooling:
      l = []
      for i, item in enumerate(inputs):
        l.append(self.denses[i](item))
      embeddings2 = tf.stack(l, axis=1)

      embeddings2 = self.pooling(embeddings2)

      embeddings = tf.concat([embeddings, embeddings2], axis=-1)
    
    # embeddings = self.fusion_dropout(embeddings)
    embedding = self.fusion(embeddings)
    embeddings = self.fusion_dropout(embeddings)

    if FLAGS.use_se:
      embedding = self.enhance(embedding)
    if FLAGS.gem_norm:
      embedding = self.gem_norm(embedding)
    if FLAGS.batch_norm:
      embedding = self.batch_norm(embedding)
    if FLAGS.layer_norm:
      embedding = self.layer_norm(embedding)

    return embedding

class Model(mt.Model):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self.vision_dense = None
    if FLAGS.continue_pretrain:
      # transformer = FLAGS.transformer.split('/')[-1]
      # continue_pretrain = f'../input/pretrain/{FLAGS.continue_version}/{transformer}/bert.h5'
      # ic(continue_pretrain)
      # self.bert.load_weights(continue_pretrain, by_name=True)

      transformer = FLAGS.transformer.split('/')[-1]
      rv = FLAGS.rv 
      if FLAGS.title_lm_only and int(FLAGS.rv) % 2 == 1:
        rv = int(FLAGS.rv) - 1
      if FLAGS.continue_version:
        rv = FLAGS.continue_version
      continue_pretrain = f'../input/pretrain/{rv}/{transformer}'
      ic(continue_pretrain)
      # with tf.name_scope("char_bert"):
      # self.bert = TFAutoModel.from_pretrained(continue_pretrain)
      self.bert = TFAutoModelForMaskedLM.from_pretrained(continue_pretrain + '/bert')
      # TODO .... currently remove head and also notice model layer names...
      self.bert = self.bert.layers[0]
      if os.path.exists(continue_pretrain + '/dense'):
        self.vision_dense = mt.load_dense(continue_pretrain + '/dense')
      # self.bert_model = mt.pretrain.bert.Model(FLAGS.transformer)
      # self.bert_model.load_weights(continue_pretrain + '/model.h5')
      # self.bert = self.bert_model.bert.layers[0]
      # self.vision_dense = bert_model.dense
      # self.continue_pretrain = continue_pretrain
      # self.loaded = False
      # tiny_count = int('tiny' in FLAGS.words_bert) + int('tiny' in FLAGS.transformer)
      # if tiny_count != 1:
      #   self.bert._name = 'char_bert'
      #   ## TODO 两个bert会因为重名 第二个_1 然后无法正常from pretrain怎么解决？
      #   # if FLAGS.words_bert == 'base':
      #   #   for layer in self.bert.layers:
      #   #     layer._name = 'char_' + layer.name 
      ic(self.bert, self.vision_dense)
    else:
      try:
        self.bert = TFAutoModel.from_pretrained(FLAGS.transformer,
                                                from_pt=FLAGS.from_pt)
      except Exception:
        self.bert = TFAutoModel.from_pretrained(FLAGS.transformer, from_pt=True)
      self.bert = self.bert.layers[0]

    self.bert.trainable = FLAGS.bert_trainable
    ic(self.bert.trainable)

    self.nextvlad = NeXtVLAD(FLAGS.frame_embedding_size,
                             FLAGS.vlad_cluster_size,
                             output_size=FLAGS.vlad_hidden_size,
                             expansion=FLAGS.vlad_expansion,
                             groups=FLAGS.vlad_groups,
                             dropout=FLAGS.vlad_dropout)    
    if FLAGS.use_merge:
      self.merge_dense = tf.keras.layers.Dense(FLAGS.frame_embedding_size)
      self.merge_dense2 = tf.keras.layers.Dense(FLAGS.frame_embedding_size)
      if not FLAGS.share_nextvlad:
        self.merge_encoder = NeXtVLAD(FLAGS.frame_embedding_size,
                              FLAGS.vlad_cluster_size,
                              output_size=FLAGS.vlad_hidden_size2,
                              expansion=FLAGS.vlad_expansion,
                              groups=FLAGS.vlad_groups,
                              dropout=FLAGS.vlad_dropout)

    if FLAGS.title_nextvlad:
      self.title_nextvlad =  NeXtVLAD(FLAGS.bert_size,
                             FLAGS.vlad_cluster_size,
                             output_size=FLAGS.vlad_hidden_size,
                             expansion=FLAGS.vlad_expansion,
                             groups=FLAGS.vlad_groups,
                             dropout=FLAGS.vlad_dropout)  

    if FLAGS.vision_dense or FLAGS.merge_vision:
      if self.vision_dense is None:
        self.vision_dense =  tf.keras.layers.Dense(FLAGS.bert_size)

    if FLAGS.title_dense:
      self.title_dense = tf.keras.layers.Dense(FLAGS.vlad_hidden_size)

    self.text_dense = tf.keras.layers.Dense(FLAGS.vlad_hidden_size)

    if FLAGS.use_vision_encoder:
      if FLAGS.vision_encoder == 'transformer':
        # self.vision_encoder = mt.layers.transformer.Encoder(6, 4,
        #               FLAGS.frame_embedding_size, FLAGS.bert_size)
        config = AutoConfig.from_pretrained('bert-base-chinese')
        config.update(
        {
          "num_hidden_layers": FLAGS.vision_layers,
          "num_attention_heads": 8,
          "vocab_size": 2,
          "hidden_size": 1536,
          "pooler_fc_size": 1536,
          "attention_probs_dropout_prob": FLAGS.vision_drop,
          "hidden_dropout_prob": FLAGS.vision_drop,
        })
        x = TFAutoModel.from_config(config)
        self.vision_encoder = x.bert
      else:
        RNN = getattr(tf.keras.layers, FLAGS.rnn)
        if FLAGS.rnn_method == 'bi':
          self.vision_encoder = tf.keras.layers.Bidirectional(RNN(int(FLAGS.frame_embedding_size / 2), return_sequences=True))
        else:
          self.vision_encoder = RNN(FLAGS.frame_embedding_size, return_sequences=True)
      self.vision_pooling = mt.layers.Pooling('att')

    if FLAGS.use_words:
      if FLAGS.segmentor == 'jieba':
        word_vocab = gezi.Vocab('../input/word_vocab.txt', FLAGS.reserve_vocab_size)
      else:
        word_vocab = gezi.Vocab('../input/sp10w.vocab', 0)
      self.word_vocab = word_vocab
      # self.word_emb = tf.keras.layers.Embedding(word_vocab.size(), FLAGS.hidden_size)
      embeddings_initializer = 'uniform' 
      hidden_size = FLAGS.hidden_size
      if FLAGS.word_w2v:
        word_emb_npy = np.load(f'../input/w2v/{FLAGS.segmentor}/{FLAGS.word_emb_size}/word.npy')
        hidden_size = word_emb_npy.shape[-1]
        if FLAGS.word_norm:
          word_emb_npy = gezi.normalize(word_emb_npy)
        embeddings_initializer = tf.keras.initializers.constant(word_emb_npy)
      self.word_emb = mt.layers.Embedding(word_vocab.size(), hidden_size, 
                                          embeddings_initializer=embeddings_initializer,
                                          trainable=FLAGS.word_trainable)

      self.words_pooling = mt.layers.Pooling(FLAGS.word_pooling)

    if not FLAGS.mdrop2:
      self.fusion = Fusion(FLAGS.hidden_size, FLAGS.se_ratio)
    else:
      self.fusions = [Fusion(FLAGS.hidden_size, FLAGS.se_ratio) for _ in range(FLAGS.mdrop_experts)]
      self.fusion = lambda x: tf.reduce_mean(tf.stack([self.fusions[i](x) for i in range(len(self.fusions))], 1), 1)
    
    if FLAGS.final_dense and util.is_pairwise():
      self.final_dense = mt.layers.MLP([*FLAGS.mlp_dims2, FLAGS.final_size], activation=mt.activation(FLAGS.activation), activate_last=FLAGS.activate_last2)
      if FLAGS.batch_norm2:
        self.bach_norm2 = tf.keras.layers.BatchNormalization()
      if FLAGS.layer_norm2:
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    else:
      self.final_dense = lambda x: x
    
    tag_vocab = gezi.Vocab('../input/tag_vocab.txt')
    self.tag_vocab = tag_vocab 
    
    Dense = tf.keras.layers.Dense 
    if FLAGS.label_strategy == 'selected_tags':
      self.num_labels = FLAGS.num_labels
      self.classifier = Dense(self.num_labels)
    elif FLAGS.label_strategy == 'all_tags':
      embeddings_initializer = 'uniform' 
      hidden_size = FLAGS.hidden_size
      if FLAGS.tag_w2v:
        # tag_emb_npy = np.load('../input/w2v/tag.npy')
        tag_emb_npy = np.load(f'../input/w2v/jieba/{FLAGS.hidden_size}/tag.npy')
        if FLAGS.tag_norm:
          tag_emb_npy = gezi.normalize(tag_emb_npy)
        hidden_size = tag_emb_npy.shape[-1]
        embeddings_initializer = tf.keras.initializers.constant(tag_emb_npy)
      self.tag_emb = mt.layers.Embedding(tag_vocab.size(), hidden_size, 
                                         embeddings_initializer=embeddings_initializer,
                                         trainable=FLAGS.tag_trainable)
      self.dots = Dot(axes=(3, 2))
      self.flatten = Flatten()
      self.remove_pred = True 
      if FLAGS.loss_fn == 'nce':
        self.tag_bias = Embedding(1, FLAGS.hidden_size)
        
    if FLAGS.parse_strategy > 2 or FLAGS.remove_pred == False:
      self.remove_pred = False
        
    if FLAGS.catloss_rate > 0:
      cat_vocab = gezi.Vocab('../input/cat_vocab.txt')
      self.cat_classifier = Dense(cat_vocab.size())
      
    if FLAGS.subcatloss_rate > 0:
      subcat_vocab = gezi.Vocab('../input/subcat_vocab.txt')
      self.subcat_classifier = Dense(subcat_vocab.size())

    self.bert_pooling = mt.layers.Pooling(FLAGS.bert_pooling)
    
    self.tag_pooling = mt.layers.Pooling(FLAGS.tag_pooling) if FLAGS.tag_pooling else None
    self.tag_pooling2 = mt.layers.Pooling(FLAGS.tag_pooling2)

    # if FLAGS.one_tower:
    #   self.denses = [tf.keras.layers.Dense(FLAGS.hidden_size) for _ in range(10)] 
    #   self.one_tower_pooling = mt.layers.Pooling(FLAGS.one_tower_pooling)

    #   # TODO classification , cardinal regression, regression? 0-1 sigmoid ?
    #   num_logits = FLAGS.num_relevances if FLAGS.one_tower_cls else 1
    #   self.one_tower_mlp = mt.layers.MLP([512, 256, num_logits], 
    #                                       activation='dice', 
    #                                       activate_last=False)
    
  def build(self, input_shape):
    if FLAGS.loss_fn == 'nce':
      self.nce_weights = self.add_weight(
        name="nce_weights",
        shape=(self.tag_vocab.size(), FLAGS.hidden_size),
        initializer=tf.keras.initializers.glorot_normal,
        trainable=True)
      self.nce_bias = self.add_weight(
        name="nce_bias",
        shape=(self.tag_vocab.size(),),
        initializer=tf.keras.initializers.glorot_normal,
        trainable=True)
    
    if FLAGS.dynamic_temperature:
      self.temperature = self.add_weight(
        name="temperature",
        shape=(),
        initializer= tf.constant_initializer(1. / FLAGS.dynamic_temperature),
        trainable=True
      )

    if FLAGS.words_bert:
      ## TODO  how to...
      # with tf.name_scope("word_bert"):
      # self.words_bert = TFAutoModel.from_pretrained(f'../input/pretrain/word/{FLAGS.rv}/{FLAGS.words_bert}/bert')
      rv = FLAGS.rv 
      if FLAGS.title_lm_only and int(FLAGS.rv) % 2 == 1:
        rv = int(FLAGS.rv) - 1
      pretrain = f'../input/pretrain/word/{rv}/{FLAGS.words_bert}/bert'
      ic(pretrain)
      self.words_bert = TFAutoModelForMaskedLM.from_pretrained(pretrain)
      self.words_bert = self.words_bert.layers[0]
      # if FLAGS.words_bert == 'base' and FLAGS.hug != 'tiny':
      #   ic(self.words_bert._name)

      self.words_bert.trainable = FLAGS.words_bert_trainable

    if FLAGS.words_rnn:
      RNN = getattr(tf.keras.layers, FLAGS.rnn)
      if FLAGS.rnn_method == 'bi':
        self.words_rnn = tf.keras.layers.Bidirectional(RNN(int(FLAGS.word_emb_size / 2), return_sequences=True))
      else:
        self.words_rnn = RNN(FLAGS.word_emb_size, return_sequences=True)

      if FLAGS.lm_target:
        vsize = self.word_vocab.size()
        self.sampled_weight = self.add_weight(name='sampled_weight',
                                              shape=(vsize, int(FLAGS.word_emb_size / 2)),
                                              #initializer = keras.initializers.RandomUniform(minval=-10, maxval=10, seed=None),
                                              dtype=tf.float32,
                                              trainable=True)

        self.sampled_bias = self.add_weight(name='sampled_bias',
                                            shape=(vsize,),
                                            #initializer = keras.initializers.RandomUniform(minval=-10, maxval=10, seed=None),
                                            dtype=tf.float32,
                                            trainable=True)

        self.softmax_loss_function = mt.seq2seq.gen_sampled_softmax_loss_function(100,
                                                                                  vsize,
                                                                                  weights=self.sampled_weight,
                                                                                  biases=self.sampled_bias,
                                                                                  log_uniform_sample=True,
                                                                                  is_predict=False,
                                                                                  sample_seed=1234)

    
    bert_layers = [self.bert]
    if FLAGS.words_bert:
      bert_layers.append(self.words_bert)
    base_layers = [layer for layer in self.layers if layer not in bert_layers]
    opt_layers = [base_layers, bert_layers]
    gezi.set('opt_layers', opt_layers)
    ic(opt_layers[-1])
          
    self.built = True
    
  def monitor_inputs(self, inputs):
    res = {}
    if 'title_ids' in inputs:
      res.update(
        {
        'title_len': tf.reduce_mean(mt.length(inputs['title_ids'])),
        'title_mean': tf.reduce_mean(tf.cast(inputs['title_ids'], mt.get_float())),
        })
    if 'pos' in inputs:
      res.update({
         'pos_mean': tf.reduce_mean(tf.cast(inputs['pos'], mt.get_float())),
         'neg_mean': tf.reduce_mean(tf.cast(inputs['neg'], mt.get_float()))
      })
    
    self.scalars(res)
    
  # def one_tower(self, embeddings_list, final_embeddings_list, inputs):
  #   l = []
  #   idx = 0
  #   for i, embeddings in enumerate(embeddings_list):
  #     l.append(final_embeddings_list[i])
  #     # for item in embeddings:
  #     #   if item.shape[-1] == FLAGS.hidden_size:
  #     #     emb = item 
  #     #   else:
  #     #     emb = self.denses[idx](item)
  #     #   l.append(emb)
  #     #   idx += 1

  #   embeddings = tf.stack(l, axis=1)
  #   # if self.first:
  #   #   tf.print('one_tower embeddings', embeddings.shape)
  #   embedding = self.one_tower_pooling(embeddings)
  #   logits = self.one_tower_mlp(embedding)
  #   return logits

  def call(self, inputs, training=None):
    add, adds = self.add, self.adds
    if isinstance(inputs, dict):
      if FLAGS.self_contrasive_rate > 0.:
        res = self.forward(inputs, training=training)
        self.final_embedding1 = self.final_embedding
        self.forward(inputs, training=training)
        self.final_embedding2 = self.final_embedding
        return res
      else:
        return self.forward(inputs, training=training)
    else:
      assert isinstance(inputs, (list, tuple))
      assert len(inputs) == 2      
      emb1 = self.forward(inputs[0], return_emb=True, index=0, training=training)
      self.final_embedding1 = emb1
      if FLAGS.self_contrasive_rate > 0.:
        self.final_embedding11 = self.forward(inputs[0], return_emb=True, index=0, training=training)
      if FLAGS.auxloss_rate > 0:
        predictions1 = self.predictions
      self.embeddings1 = copy.copy(self.embeddings)
      emb2 = self.forward(inputs[1], return_emb=True, index=1, training=training)
      self.final_embedding2 = emb2
      if FLAGS.self_contrasive_rate > 0.:
        self.final_embedding22 = self.forward(inputs[1], return_emb=True, index=1, training=training)
      if FLAGS.auxloss_rate > 0:
        predictions2 = self.predictions
      self.embeddings2 = copy.copy(self.embeddings)
      self.dot_sim = None
      # if not FLAGS.one_tower:
      similarity = mt.element_wise_cosine(emb1, emb2)
      # else:
      #   # 注意这个方案简单 但是不能反映相同数值 相同rank 使用one_tower_cls 是可以的 但是需要ft_loss_fn 不使用 corr优化？  可以对比一下。。 TODO
      #   logits = self.one_tower([self.embeddings1, self.embeddings2], [self.final_embedding1, self.final_embedding2], inputs)
      #   if not FLAGS.one_tower_cls:
      #     similarity = tf.nn.sigmoid(logits)
      #     # tf.print('------------', logits, similarity, inputs[0]['relevance'])
      #     if FLAGS.from_logits:
      #       self.dot_sim = logits
      #   else:
      #     similarity = tf.argmax(logits, axis=-1)
         
      if FLAGS.from_logits and self.dot_sim is None:
        self.dot_sim = mt.element_wise_cosine(emb1, emb2) 
      
      if FLAGS.auxloss_rate > 0:
        aux_loss_fn = self.get_loss_fn(parse_strategy=1, label_strategy='all_tags', loss_fn_name='multi')
        label1 = tf.concat([tf.ones_like(inputs[0]['pos']), tf.zeros_like(inputs[0]['neg'])], -1)
        label2 = tf.concat([tf.ones_like(inputs[1]['pos']), tf.zeros_like(inputs[1]['neg'])], -1)
        aux_loss = (aux_loss_fn(label1, predictions1) + aux_loss_fn(label2, predictions2)) 
        self.scalar('loss/aux', aux_loss)
        self.aux_loss = aux_loss * FLAGS.auxloss_rate

      # tf.print(similarity)
      return similarity
      
  def forward(self, inputs, return_emb=False, index=None, training=None):
    add, adds = self.add, self.adds
    self.clear()

    self.inputs = inputs
    if training:
      self.monitor_inputs(inputs)

    embeddings = self.feats

    vision_feats = inputs['frames']
    num_frames = tf.reshape(inputs['num_frames'], [-1,])

    if FLAGS.use_words:
      word_ids = inputs['word_ids'] if 'word_ids' in  inputs else inputs['title_word_ids']
      self.word_ids = word_ids
      words_embeddings = self.word_emb(word_ids)
      if FLAGS.words_rnn:
        words_embeddings = self.words_rnn(words_embeddings)
        if FLAGS.lm_target:
          return words_embeddings
      words_embedding = self.words_pooling(words_embeddings, mt.length(word_ids))
      add(words_embedding, 'words')

    if FLAGS.words_bert:
      if 'word_ids' in inputs:
        input_ids = inputs['word_ids']
      else:
        input_ids = inputs['title_word_ids']
      if FLAGS.max_len:
        input_ids = input_ids[:,:FLAGS.max_len]

      if FLAGS.segmentor == 'jieba':
        mask = tf.cast(input_ids < 100000, input_ids.dtype)
        input_ids = input_ids * mask + tf.ones_like(mask) * (1 - mask)
      words_embeddings = self.words_bert([input_ids])[0]
      words_embedding = self.bert_pooling(words_embeddings)
      add(words_embedding, 'words_bert')

    if FLAGS.use_vision or FLAGS.use_merge:
      vision_embedding = self.nextvlad([vision_feats, num_frames])
      vision_embs = self.nextvlad.feats
      vision_embedding = vision_embedding * \
          tf.cast(tf.expand_dims(num_frames, -1) > 0, vision_embedding.dtype)
      self.vision_embedding = self.vision_dense(vision_embedding) if FLAGS.vision_dense else vision_embedding
  
    if FLAGS.use_title or FLAGS.use_merge:
      if 'input_ids' in inputs:
        input_ids, attention_mask, token_type_ids = inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids']
        if FLAGS.max_len:
          input_ids, attention_mask, token_type_ids = input_ids[:,:FLAGS.max_len], attention_mask[:,:FLAGS.max_len], token_type_ids[:,:FLAGS.max_len]
        title_out = self.bert([input_ids, attention_mask, token_type_ids])
      else:
        input_ids = inputs['title_ids']
        title_out = self.bert([inputs['title_ids'], inputs['title_mask']])
      title_embedding = self.bert_pooling(title_out[0])
      self.title_embedding = title_embedding

      if FLAGS.merge_vision:
        # https://huggingface.co/transformers/_modules/transformers/modeling_tf_bert.html
        input_ids = inputs['title_ids'] if not FLAGS.max_len else inputs['title_ids'][:,:FLAGS.max_len]
        attention_mask = inputs['title_mask'] if not FLAGS.max_len else inputs['title_mask'][:,:FLAGS.max_len]
        # title_embs = tf.gather(self.bert.bert.embeddings.weight, input_ids)
        title_embs = tf.gather(self.bert.embeddings.weight, input_ids)
        if not FLAGS.merge_vision_after_vlad:
          vision_embs = self.vision_dense(vision_feats)  
          attention_mask = tf.concat([attention_mask, tf.sequence_mask(num_frames, FLAGS.max_frames, dtype=attention_mask.dtype)], 1)
        else:
          vision_embs = self.vision_dense(vision_embs)
          attention_mask = tf.concat([attention_mask, tf.ones_like(vision_embs[:,:,0], dtype=attention_mask.dtype)], 1)
        input_embs = tf.concat([title_embs,vision_embs], 1)
        l =  [
              tf.zeros_like(input_ids), 
              tf.cast(tf.fill([mt.get_shape(vision_embs, 0), 
                      mt.get_shape(vision_embs, 1)],
                        1), input_ids.dtype)
            ]
        token_type_ids = tf.concat(l, axis=-1)
        title_out2 = self.bert(None, attention_mask, token_type_ids, inputs_embeds=input_embs)
        title_embedding2 = self.bert_pooling(title_out2[0])
        add(title_embedding2, 'merge_vision')

    if FLAGS.title_nextvlad:
      title_nextvlad = self.title_nextvlad([title_out[0], mt.length(input_ids)])
      add(title_nextvlad, 'title_nextvlad')

    if FLAGS.use_asr:
      asr_out = self.bert(inputs['asr_ids'][:,:FLAGS.max_asr_len], inputs['asr_mask'][:,:FLAGS.max_asr_len])
      asr_embedding = self.bert_pooling(asr_out[0])
      self.asr_embedding = asr_embedding

    if FLAGS.use_first_vision:
      add(vision_feats[:,0], 'first_vsion')

    if FLAGS.use_vision_encoder:
      # vision_feats = self.vision_encoder(vision_feats, num_frames)
      if FLAGS.vision_encoder == 'transformer':
        vision_feats2 = self.vision_encoder(None, tf.sequence_mask(num_frames), inputs_embeds=vision_feats)[0]
      else:
        ## TODO not work
        # vision_feats2 = self.vision_encoder(vision_feats, tf.sequence_mask(num_frames))
        vision_feats2 = self.vision_encoder(vision_feats)  
      vision_emb = self.vision_pooling(vision_feats2, num_frames)
      add(vision_emb, 'vision_encoder')
    if FLAGS.use_merge:
      merge_encoder = self.merge_encoder if not FLAGS.share_nextvlad else self.nextvlad
      if FLAGS.merge_method == 0:
        merge_feats = [self.merge_dense(title_out[0]), vision_feats]
        merge_len = title_out[0].shape[1] + num_frames
      elif FLAGS.merge_method == 1:
        merge_feats = [self.merge_dense2(words_embeddings), vision_feats]
        merge_len = words_embeddings.shape[1] + num_frames
      elif FLAGS.merge_method == 2:
        merge_feats = [self.merge_dense(title_out[0]), self.merge_dense2(words_embeddings), vision_feats]
        merge_len = title_out[0].shape[1] + words_embeddings.shape[1] + num_frames
      elif FLAGS.merge_method == 3:
        assert FLAGS.merge_vision
        merge_feats = [title_out2[0]]
        merge_len = title_out[0].shape[1] + num_frames
      elif FLAGS.merge_method == 4:
        assert FLAGS.use_vision_encoder
        merge_feats = [vision_feats2]
        merge_len = num_frames
      elif FLAGS.merge_method == 5:
        assert FLAGS.use_vision_encoder
        merge_feats = [self.merge_dense(title_out[0]), vision_feats2]
        merge_len = title_out[0].shape[1] + num_frames
      merge_embedding = merge_encoder([tf.concat(merge_feats, 1), merge_len]) 
      self.merge_embedding = merge_embedding

    if FLAGS.use_vision:
      add(self.vision_embedding, 'vision')
      self.monitor_emb(vision_embedding, 'vision_emb')
      
    if FLAGS.use_title:
      add(self.title_embedding, 'title')
      self.monitor_emb(title_embedding, 'title_emb')

    if FLAGS.use_words:
      text_embedding = tf.concat([title_embedding, words_embedding], -1)
      self.text_embedding = self.text_dense(text_embedding)
      
    if FLAGS.use_asr:
      add(self.asr_embedding, 'asr')
      self.monitor_emb(asr_embedding, 'asr_emb')

    if FLAGS.use_merge:
      add(merge_embedding, 'merge')
      self.monitor_emb(merge_embedding, 'merge_emb')
    
    self.embeddings = embeddings
    self.print_feats(logging.ice)
    assert(len(embeddings))
    final_embedding = self.fusion(embeddings)
 
    if FLAGS.tag_softmax:
      # tf.print(self.tag_emb.embeddings)
      # tf.print(self.tag_emb.get_weights())
      # tf.print(self.tag_emb.weights)
      # exit(0)
      # weights = tf.nn.softmax(mt.dot(final_embedding, self.tag_emb.get_weights()))
      # self.final_embedding = tf.matmul(weights, self.tag_emb.get_weights())
      # weights = tf.nn.softmax(mt.dot(final_embedding, self.tag_emb.embeddings))
      # self.final_embedding = tf.matmul(weights, self.tag_emb.embeddings)
      weights = tf.nn.softmax(mt.dot(final_embedding, self.tag_emb(None)))
      final_embedding = tf.matmul(weights, self.tag_emb(None))
    
    self.final_embedding = self.final_dense(final_embedding)
    if util.is_pairwise():
      if FLAGS.batch_norm2:
        self.final_embedding = self.batch_norm(self.final_embedding)
      if FLAGS.layer_norm2:
        self.final_embedding = self.layer_norm(self.final_embedding)
      
    if FLAGS.top_tags and not training:
      if FLAGS.l2_norm:
         tag_sim = mt.dot(tf.nn.l2_normalize(final_embedding), tf.nn.l2_normalize(self.tag_emb(None)))
      else:
        tag_sim = mt.dot(final_embedding, self.tag_emb(None))
      # tag_sim = mt.dot(tf.nn.l2_normalize(self.final_embedding), tf.nn.l2_normalize(self.tag_emb(None)))
      # self.top_tags = tf.argsort(-tag_sim)[:, :FLAGS.top_tags]
      self.top_weights, self.top_tags = tf.nn.top_k(tag_sim, k=FLAGS.top_tags)
      if index == 0:
        self.top_weights1, self.top_tags1 = self.top_weights, self.top_tags
      if index == 1:
        self.top_weights2, self.top_tags2 = self.top_weights, self.top_tags
      
    self.monitor_emb(self.final_embedding, 'final_emb', zero_ratio=True)
    self.monitor_emb(self.tag_emb(None), 'tag_emb', zero_ratio=True)
    
    if return_emb and FLAGS.auxloss_rate == 0:
      return self.final_embedding
    
    if FLAGS.loss_fn == 'nce':
      return inputs['y']
    
    if FLAGS.remove_pred and not training:
      return tf.ones_like(inputs['vid'], dtype=mt.get_float())
    
    try:
      if FLAGS.label_strategy == 'selected_tags':
        predictions = self.classifier(final_embedding)
        if not FLAGS.from_logits:
          predictions = tf.nn.sigmoid(predictions)
      elif FLAGS.label_strategy == 'all_tags':
        vid_emb = tf.expand_dims(final_embedding, 1)
        pos_emb = self.tag_emb(inputs['pos'])
        if FLAGS.tag_pooling:
          pos_emb = self.tag_pooling(pos_emb, mt.length(inputs['pos']))
          pos_emb = tf.expand_dims(pos_emb, 1)
        neg_emb = self.tag_emb(inputs['neg'])
        context_emb = tf.concat([pos_emb, neg_emb], 1)
        context_emb = tf.expand_dims(context_emb, 2)
        # bs, 1, dim    bs, n, 1, dim

        if FLAGS.l2_norm:
          context_emb = tf.nn.l2_normalize(context_emb, -1)
          vid_emb = tf.nn.l2_normalize(vid_emb, -1)
          
        dots = self.dots([context_emb, vid_emb])
        predictions = self.flatten(dots)

        if not FLAGS.from_logits:
          if not FLAGS.l2_norm:
            if FLAGS.tag_pooling:
              predictions = tf.nn.softmax(predictions, -1)
            else:
              predictions = tf.nn.sigmoid(predictions)
          else:
            # predictions = (predictions + 1.) / 2.
            predictions = tf.maximum(predictions, 0.)
        else:
          if FLAGS.dynamic_temperature:
            predictions /= self.temperature
          else:
            # by defualt * 1.
            predictions *= FLAGS.temperature
      else:
        raise ValueError(FLAGS.label_strategy)

      self.predictions = predictions
      
      if return_emb:
        return self.final_embedding
      else:
        return self.predictions
    except Exception as e:
      ic(e)
      return tf.ones_like(inputs['vid'], dtype=mt.get_float())
  
  def contrasive_loss(self):    
    perms = []
    for _ in range(FLAGS.num_vids):  
      perm = tf.random.shuffle(tf.range(tf.shape(self.inputs['vid'])[0]))
      perms.append(perm)
    
    texts = [self.text_embedding]
    for perm in perms:
      text = tf.gather(self.text_embedding, perm, axis=0)
      texts.append(text)
    texts = tf.stack(texts, 1)
    
    # bs, 1, hidden_size
    emb = tf.expand_dims(self.vision_embedding, 1)
    # bs, vids, 1, hidden_size
    embs = tf.expand_dims(texts, 2)
    # if FLAGS.contrasive_norm:
    #   emb = tf.nn.l2_normalize(emb, -1)
    #   embs = tf.nn.l2_normalize(embs, -1)

    # bs, vids
    emb_sim = self.flatten(self.dots([embs, emb]))
    
    loss_obj = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE,
                                                      from_logits=True)       
    
    bs = mt.get_shape(emb_sim, 0)
    label = tf.repeat(tf.constant([[1] + [0] * FLAGS.num_vids]), bs, 0)

    loss = loss_obj(label, emb_sim)
    
    loss *= FLAGS.loss_scale 
    loss = mt.reduce_over(loss)
    return loss

  def self_contrasive_loss(self, embedding1, embedding2):
    # tf.print(embedding1.shape, embedding2.shape)
    embeddings = [embedding1, embedding2]    
    perms = []
    for _ in range(FLAGS.num_vids):  
      perm = tf.random.shuffle(tf.range(tf.shape(embedding1)[0]))
      perms.append(perm)
    
    embs = [embedding2]
    for i, perm in enumerate(perms):
      embedding = embeddings[i % 2]
      emb = tf.gather(embedding, perm, axis=0)
      embs.append(emb)
    embs = tf.stack(embs, 1)
    
    # bs, 1, hidden_size
    emb = tf.expand_dims(embedding1, 1)
    # bs, vids, 1, hidden_size
    embs = tf.expand_dims(embs, 2)

    # bs, vids
    emb_sim = self.flatten(self.dots([embs, emb]))
    
    loss_obj = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE,
                                                      from_logits=True)       
    
    bs = mt.get_shape(emb_sim, 0)
    label = tf.repeat(tf.constant([[1] + [0] * FLAGS.num_vids]), bs, 0)

    loss = loss_obj(label, emb_sim)
    
    loss *= FLAGS.loss_scale 
    loss = mt.reduce_over(loss)
    return loss
  
  def get_loss_fn(self, parse_strategy=None, label_strategy=None, loss_fn_name=None):
    parse_strategy = parse_strategy or FLAGS.parse_strategy
    label_strategy = label_strategy or FLAGS.label_strategy
    loss_fn_name = loss_fn_name or FLAGS.loss_fn
    def loss_fn(y_true, y_pred):
      if FLAGS.lm_target:
        return mt.losses.sampled_bilm_loss(self.word_ids, y_pred, self.softmax_loss_function)

      y_true = tf.cast(y_true, tf.float32)
      if FLAGS.numerical_stable:
        y_true = tf.maximum(y_true, 0.01)
        y_true = tf.minimum(y_true, 0.99)
      y_pred = tf.cast(y_pred, tf.float32)
      # pairwise      
      if parse_strategy > 2:

        if FLAGS.pred_adjust == 1:
          y_pred = tf.maximum(y_pred, 0.)
        elif FLAGS.pred_adjust == 2:
          y_pred = (1. + y_pred) / 2.

        if FLAGS.loss_fn == 'mse':
          loss_name = 'mse'
          loss_obj = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
          loss = loss_obj(y_true, y_pred)
        elif FLAGS.loss_fn == 'mae':
          loss_name = 'mae'
          loss_obj = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
          loss = loss_obj(y_true, y_pred)
        elif FLAGS.loss_fn == 'msle':
          loss_name = 'msle'
          loss_obj = tf.keras.losses.MeanSquaredLogarithmicError(reduction=tf.keras.losses.Reduction.NONE)
          loss = loss_obj(y_true, y_pred)
        elif FLAGS.loss_fn in ['correlation', 'corr', 'corr_mse', 'corr_sigmoid']:
          loss_name = 'corr'
          loss = 1. -  tfp.stats.correlation(y_true, y_pred, sample_axis=0, event_axis=None)
          # (bs, 1) (bs, 1) (bs, 1)
          # tf.print(y_true.shape, y_pred.shape, loss.shape)
          bs = mt.get_shape(y_true, 0)
          loss = tf.repeat(loss, bs, 0)
          # （bs)
          # tf.print(loss.shape)
          if FLAGS.loss_fn == 'corr_mse':
            loss_name = FLAGS.loss_fn
            loss *= 0.01
            self.scalar('loss/corr', mt.reduce_over(loss))
            loss2 = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)(y_true, y_pred)
            self.scalar('loss/mse', mt.reduce_over(loss2))
            loss += loss2
          elif FLAGS.loss_fn == 'corr_sigmoid':
            loss_name = FLAGS.loss_fn
            loss *= 0.1
            self.scalar('loss/corr', mt.reduce_over(loss))
            loss2 = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE,
                                                       from_logits=False)(y_true, tf.math.maximum(y_pred, 0.))
            self.scalar('loss/sigmoid', mt.reduce_over(loss2))
            loss += loss2

        elif FLAGS.loss_fn in ['spearmanr', 'spear', 'sp']:
          loss_name = 'spearmanr'
          from fast_soft_sort.tf_ops import soft_rank, soft_sort
          y_true = soft_rank(y_true)
          y_pred = soft_rank(y_pred)
          loss = 1. -  tfp.stats.correlation(y_true, y_pred, sample_axis=0, event_axis=None)
          bs = mt.get_shape(y_true, 0)
          loss = tf.repeat(loss, bs, 0)

        elif FLAGS.loss_fn == 'pair':
          loss_name = 'pair_mse'
          loss = tf.compat.v1.losses.mean_pairwise_squared_error(y_true, y_pred)
          bs = mt.get_shape(y_true, 0)
          loss = tf.repeat(loss, bs, 0)

        elif FLAGS.loss_fn == 'auc':
          loss_name = 'auc'
          loss = global_objectives.precision_recall_auc_loss(y_true, y_pred, 
                              reuse=tf.compat.v1.AUTO_REUSE, scope='auc_loss')[0] 

        else:
          loss_name = 'sigmoid'
          y_pred = tf.maximum(y_pred, 0.)
          # from_logits = FLAGS.from_logits
          from_logits = False
          loss_obj = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE,
                                                        from_logits=from_logits)
          # if from_logits:
          #   loss = loss_obj(y_true, self.dot_sim)
          # else:
          #   loss = loss_obj(y_true, tf.maximum(y_pred, 0.))
          loss = loss_obj(y_true, y_pred)

        # mask = tf.cast(y_true == 1, tf.float32)
        # weight = tf.ones_like(loss) * (1 - mask) + tf.ones_like(loss) * FLAGS.pos_weight * mask
        # loss *= weight
        # tf.print('------1', loss, loss.shape)
        if FLAGS.weight_loss:
          ## similar just fro reproduce
          if FLAGS.weight_method == 0:
            weight = tf.math.log(self.inputs['label'] * 100. + 2.) ** FLAGS.weight_power
          else:
            weight = tf.math.log(self.inputs['label'] * 100)
          # weight = y_true + 1.
          loss *= weight

        if FLAGS.ft_ext and FLAGS.ext_weight:
          # loss *= (self.inputs['type_weight'] * self.inputs['relevance_weight'])
          loss *= self.inputs['type_weight'] 

        # tf.print('------2', loss, loss.shape)
        loss *= FLAGS.loss_scale
        loss = mt.reduce_over(loss)
        self.scalar(f'loss/pairwise/{loss_name}', loss)
        if FLAGS.auxloss_rate > 0.:
          loss += self.aux_loss
        return loss
         
      loss_ = 0.
      if FLAGS.normalloss_rate > 0.:
        if label_strategy == 'selected_tags':
          from_logits = False if FLAGS.final_activation else True
          loss_obj = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE,
                                                        from_logits=from_logits)
          loss = loss_obj(y_true, y_pred)
          # # 因为很稀疏 大部分类别loss为0几乎 所以按类别sum算loss 效果更好 否则loss太小
          # # 另外似乎较大的loss 几百 几千 更加数值稳定 相对于小的loss
          if FLAGS.loss_sum_byclass:
            loss *= y_true.shape[-1]
        elif label_strategy == 'all_tags':
          mask = tf.concat([tf.cast(self.inputs['pos'] > 1, tf.float32), tf.cast(self.inputs['neg'] > 1, tf.float32)], -1)
          if loss_fn_name == 'multi':
            loss_obj = tf.keras.losses.CategoricalCrossentropy(from_logits=FLAGS.from_logits, 
                                                               reduction=tf.keras.losses.Reduction.NONE)
            y_true_ = tf.concat([tf.ones_like(y_pred[:,:1]),
                                tf.zeros_like(y_pred[:,-FLAGS.num_negs:])], -1)
            tags = FLAGS.loss_tags or FLAGS.max_tags
            tags = min(tags, y_pred.shape[-1] - FLAGS.num_negs)
            if FLAGS.tag_pooling:
              tags = 1
            losses = []
            y_pred_neg = y_pred[:,-FLAGS.num_negs:]
            for i in range(tags):
              y_pred_ = tf.concat([y_pred[:,i:i+1], y_pred_neg], -1)
              # loss = loss_obj(y_true_, y_pred_)
              # self.scalar(f'loss/softmax', mt.reduce_over(loss))
              if FLAGS.arc_face:
                y_pred_ = mt.losses.arc_margin_product(y_pred_, y_true_)
                # loss2 = loss_obj(y_true_, y_pred_)
                # self.scalar(f'loss/arcface', mt.reduce_over(loss2))
                # loss += loss2
              loss = loss_obj(y_true_, y_pred_)
              loss *= mask[:,i]
              losses.append(loss)
            reduce_fn = tf.reduce_mean if not FLAGS.hard_tag else tf.reduce_max
            loss = reduce_fn(tf.stack(losses, 1), axis=-1)
            loss *= FLAGS.loss_scale
          elif loss_fn_name == 'binary':
            loss_obj = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE,
                                                          from_logits=FLAGS.from_logits)
            y_true = tf.reshape(y_true, [-1, (FLAGS.max_tags + FLAGS.num_negs)])
            if FLAGS.from_logits and FLAGS.from_logits_mask > 0:
              mask = mask + (mask - 1) * FLAGS.from_logits_mask
              y_pred *= mask
            loss = loss_obj(y_true, y_pred)
            loss *= FLAGS.loss_scale
          elif loss_fn_name == 'nce':
            def loss_fn(y_true, y_pred):
              loss  = tf.nn.nce_loss(
                self.nce_weights, self.nce_bias, self.inputs['pos'], self.final_embedding, FLAGS.num_negs, self.tag_vocab.size(), num_true=FLAGS.max_tags,
                sampled_values=None, remove_accidental_hits=False, name='nce_loss'
              )
              loss *= FLAGS.loss_scale
        else:
          raise ValueError(FLAGS.label_strategy)
      
        loss = mt.reduce_over(loss)
        self.scalar('loss/normal', loss)
        loss_ += FLAGS.normalloss_rate * loss
        
      if FLAGS.catloss_rate > 0:
        from_logits = False if FLAGS.final_activation else True
        loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=FLAGS.from_logits, 
                                                                  reduction=tf.keras.losses.Reduction.NONE)
        label = self.inputs['cat']
        pred = self.cat_classifier(self.final_embedding)
        if not FLAGS.from_logits:
          pred = tf.nn.sigmoid(pred)
        loss = loss_obj(label, pred)
        loss *= tf.reshape(tf.cast(label > 1, tf.float32), loss.shape)
        loss *= FLAGS.loss_scale
        loss = mt.reduce_over(loss)
        self.scalar('loss/cat', loss)
        loss_ += FLAGS.catloss_rate * loss
        
      if FLAGS.subcatloss_rate > 0:
        from_logits = False if FLAGS.final_activation else True
        loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=FLAGS.from_logits, 
                                                                  reduction=tf.keras.losses.Reduction.NONE)
        label = self.inputs['subcat']
        pred = self.subcat_classifier(self.final_embedding)
        if not FLAGS.from_logits:
          pred = tf.nn.sigmoid(pred)
        loss = loss_obj(label, pred)
        loss *= tf.reshape(tf.cast(label > 1, tf.float32), loss.shape)
        loss *= FLAGS.loss_scale
        loss = mt.reduce_over(loss)
        self.scalar('loss/subcat', loss)
        loss_ += FLAGS.subcatloss_rate * loss
        
      if FLAGS.contrasive_rate > 0:
        loss = self.contrasive_loss()
        self.scalar('loss/contrasive', loss)
        loss_ += FLAGS.contrasive_rate * loss
        
      if FLAGS.zeroloss_rate > 0:
        loss = tf.reduce_sum(tf.cast(self.final_embedding == 0, tf.float32), axis=-1)
        loss = mt.reduce_over(loss)
        self.scalar('loss/zero', loss)
        loss_ += FLAGS.zeroloss_rate * loss
      
      if FLAGS.self_contrasive_rate > 0:
        if util.is_pointwise():
          loss = self.self_contrasive_loss(self.final_embedding1, self.final_embedding2)
        else:
          loss1 = self.self_contrasive_loss(self.final_embedding1, self.final_embedding11)
          loss2 = self.self_contrasive_loss(self.final_embedding2, self.final_embedding22)
          loss = (loss1 + loss2) / 2.
        self.scalar('loss/self_contrasive', loss)
        loss_ += FLAGS.self_contrasive_rate * loss

      return loss_
    
    return loss_fn
     
