
import copy
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.keras.layers import Dot, Embedding, Flatten

from transformers import TFAutoModel, TFAutoModelForMaskedLM, TFAutoModelForSequenceClassification
from transformers import AutoConfig, BertConfig, TFBertModel

import gezi
import melt as mt
from ..config import *
from .. import util

# model.py 用于之前模型迭代 可能有较多if else 很多分支不会走到
# model2.py 是当前迭代最优模型 并且去掉大部分未走到的实验分支(确定无效的)
# words.py 在model2基础上 改为使用word bert 

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

    # for cluster weights
    self.cluster_dense1 = tf.keras.layers.Dense(self.groups * self.cluster_size,
                                                activation=None,
                                                use_bias=False)
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

# TODO change to Fusion
class Fusion(tf.keras.layers.Layer):
  def __init__(self, hidden_size, se_ratio, **kwargs):
    super().__init__(**kwargs)
    ic(mt.activation(FLAGS.activation))

    self.fusion = mt.layers.MultiDropout(dims=[*FLAGS.mlp_dims, hidden_size], 
                                          activation=mt.activation(FLAGS.activation), 
                                          drop_rate=FLAGS.mdrop_rate,
                                          num_experts=FLAGS.mdrop_experts,
                                          activate_last=FLAGS.activate_last)
    self.fusion_dropout = tf.keras.layers.Dropout(FLAGS.fusion_dropout)                                    
    self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
  def call(self, inputs, **kwargs):
    embeddings = tf.concat(inputs, axis=1)
    # embeddings = self.fusion_dropout(embeddings)
    embedding = self.fusion(embeddings)
    embeddings = self.fusion_dropout(embeddings)
    embedding = self.layer_norm(embedding)

    return embedding

class Model(mt.Model):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self.vision_dense = None

    self.nextvlad = NeXtVLAD(FLAGS.frame_embedding_size,
                             FLAGS.vlad_cluster_size,
                             output_size=FLAGS.vlad_hidden_size,
                             expansion=FLAGS.vlad_expansion,
                             groups=FLAGS.vlad_groups,
                             dropout=FLAGS.vlad_dropout)    

    self.merge_dense = tf.keras.layers.Dense(FLAGS.frame_embedding_size)
    self.merge_dense2 = tf.keras.layers.Dense(FLAGS.frame_embedding_size)
    if not FLAGS.share_nextvlad:
      self.merge_encoder = NeXtVLAD(FLAGS.frame_embedding_size,
                            FLAGS.vlad_cluster_size,
                            output_size=FLAGS.vlad_hidden_size2,
                            expansion=FLAGS.vlad_expansion,
                            groups=FLAGS.vlad_groups,
                            dropout=FLAGS.vlad_dropout)

    self.title_nextvlad =  NeXtVLAD(FLAGS.bert_size,
                            FLAGS.vlad_cluster_size,
                            output_size=FLAGS.vlad_hidden_size,
                            expansion=FLAGS.vlad_expansion,
                            groups=FLAGS.vlad_groups,
                            dropout=FLAGS.vlad_dropout)  

    if self.vision_dense is None:
      self.vision_dense =  tf.keras.layers.Dense(FLAGS.bert_size)

    self.text_dense = tf.keras.layers.Dense(FLAGS.vlad_hidden_size)

    self.fusion = Fusion(FLAGS.hidden_size, FLAGS.se_ratio)
    
    tag_vocab = gezi.Vocab('../input/tag_vocab.txt')
    self.tag_vocab = tag_vocab 
    
    Dense = tf.keras.layers.Dense 

    # all_tags much better
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
    if FLAGS.parse_strategy > 2 or FLAGS.remove_pred == False:
      self.remove_pred = False
        
    self.bert_pooling = mt.layers.Pooling(FLAGS.bert_pooling)

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
    
  def build(self, input_shape):  
    # not used 
    if FLAGS.dynamic_temperature:
      self.temperature = self.add_weight(
        name="temperature",
        shape=(),
        initializer= tf.constant_initializer(1. / FLAGS.dynamic_temperature),
        trainable=True
      )

    rv = FLAGS.rv 
    if FLAGS.title_lm_only and int(FLAGS.rv) % 2 == 1:
      rv = int(FLAGS.rv) - 1
    pretrain = f'../input/pretrain/word/{rv}/{FLAGS.words_bert}/bert'
    ic(pretrain)
    self.words_bert = TFAutoModelForMaskedLM.from_pretrained(pretrain)
    self.words_bert = self.words_bert.layers[0]
    self.words_bert.trainable = FLAGS.words_bert_trainable
    self.bert = self.words_bert

    bert_layers = [self.bert]
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
    

  def call(self, inputs, training=None):
    if isinstance(inputs, dict):
      # pointwise
      return self.forward(inputs, training=training)
    else:
      # pairwise
      assert isinstance(inputs, (list, tuple))
      assert len(inputs) == 2      
      emb1 = self.forward(inputs[0], return_emb=True, index=0, training=training)
      self.final_embedding1 = emb1
      self.embeddings1 = copy.copy(self.embeddings)
      emb2 = self.forward(inputs[1], return_emb=True, index=1, training=training)
      self.final_embedding2 = emb2
      self.embeddings2 = copy.copy(self.embeddings)
      self.dot_sim = None
      similarity = mt.element_wise_cosine(emb1, emb2)
         
      if FLAGS.from_logits and self.dot_sim is None:
        self.dot_sim = mt.element_wise_cosine(emb1, emb2) 
      
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

    vision_embedding = self.nextvlad([vision_feats, num_frames])
    vision_embs = self.nextvlad.feats
    vision_embedding = vision_embedding * \
        tf.cast(tf.expand_dims(num_frames, -1) > 0, vision_embedding.dtype)
    self.vision_embedding = vision_embedding
    add(self.vision_embedding, 'vision')
    self.monitor_emb(vision_embedding, 'vision_emb')

    # merge vision better
    merge_encoder = self.merge_encoder 
    if not FLAGS.merge_vision:
      if 'word_ids' in inputs:
        input_ids = inputs['word_ids']
      else:
        input_ids = inputs['title_word_ids']
      title_out = self.bert([input_ids])
      title_embedding = self.bert_pooling(title_out[0])
      self.title_embedding = title_embedding
      add(title_embedding, 'title')

      merge_feats = [self.merge_dense(title_out[0]), vision_feats]
      merge_len = title_out[0].shape[1] + num_frames
    else:
      # https://huggingface.co/transformers/_modules/transformers/modeling_tf_bert.html
      input_ids = inputs['title_word_ids'] if not FLAGS.max_len else inputs['title_word_ids'][:,:FLAGS.max_len]
      attention_mask = tf.cast(input_ids > 0, input_ids.dtype)
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
      title_vision_out = self.bert(None, attention_mask, token_type_ids, inputs_embeds=input_embs)
      title_vision_embedding = self.bert_pooling(title_vision_out[0])
      add(title_vision_embedding, 'title_vision')

      merge_feats = [title_vision_out[0]]
      merge_len = title_vision_out[0].shape[1] + num_frames

    merge_embedding = merge_encoder([tf.concat(merge_feats, 1), merge_len]) 
    self.merge_embedding = merge_embedding
    add(merge_embedding, 'merge')
    self.monitor_emb(merge_embedding, 'merge_emb')
    
    self.embeddings = embeddings
    self.print_feats(logging.ice)
    assert(len(embeddings))
    final_embedding = self.fusion(embeddings)   
    self.final_embedding = final_embedding
      
    if FLAGS.top_tags and not training:
      if FLAGS.l2_norm:
         tag_sim = mt.dot(tf.nn.l2_normalize(final_embedding), tf.nn.l2_normalize(self.tag_emb(None)))
      else:
        tag_sim = mt.dot(final_embedding, self.tag_emb(None))

      self.top_weights, self.top_tags = tf.nn.top_k(tag_sim, k=FLAGS.top_tags)
      if index == 0:
        self.top_weights1, self.top_tags1 = self.top_weights, self.top_tags
      if index == 1:
        self.top_weights2, self.top_tags2 = self.top_weights, self.top_tags
      
    self.monitor_emb(self.final_embedding, 'final_emb', zero_ratio=True)
    self.monitor_emb(self.tag_emb(None), 'tag_emb', zero_ratio=True)
    
    if return_emb and FLAGS.auxloss_rate == 0:
      return self.final_embedding
    
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
  
  def pairwise_loss(self, y_true, y_pred):
    loss_name = 'mse'
    loss_obj = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    loss = loss_obj(y_true, y_pred)

    if FLAGS.weight_loss:
      ## similar just for reproduce
      if FLAGS.weight_method == 0:
        weight = tf.math.log(self.inputs['label'] * 100. + 2.) ** FLAGS.weight_power
      else:
        weight = tf.math.log(self.inputs['label'] * 100)
      # weight = y_true + 1.
      loss *= weight

    loss *= FLAGS.loss_scale
    loss = mt.reduce_over(loss)
    self.scalar(f'loss/pairwise/{loss_name}', loss)
    if FLAGS.auxloss_rate > 0.:
      loss += self.aux_loss
    return loss

  def pointwise_loss(self, y_true, y_pred, label_strategy=None, loss_fn_name=None):
    loss_ = 0.

    if FLAGS.normalloss_rate > 0:
      # selected tags只为了和baseline对比
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
          # pointwise 走这里实际！multi much better then binary
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
      else:
        raise ValueError(label_strategy)
      
      loss = mt.reduce_over(loss)
      self.scalar('loss/normal', loss)
      loss_ += FLAGS.normalloss_rate * loss
    
    return loss_

  def get_loss_fn(self, parse_strategy=None, label_strategy=None, loss_fn_name=None):
    parse_strategy = parse_strategy or FLAGS.parse_strategy
    label_strategy = label_strategy or FLAGS.label_strategy
    loss_fn_name = loss_fn_name or FLAGS.loss_fn
    def loss_fn(y_true, y_pred):
      y_true = tf.cast(y_true, tf.float32)
      y_pred = tf.cast(y_pred, tf.float32)
      # pairwise      
      if parse_strategy > 2:
        return self.pairwise_loss(y_true, y_pred)
      else:
        return self.pointwise_loss(y_true, y_pred, label_strategy, loss_fn_name)
    return loss_fn
     
