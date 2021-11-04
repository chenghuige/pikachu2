#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   eval.py
#        \author   chenghuige
#          \date   2021-01-09 17:51:06.853603
#   \Description
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from tensorflow.python.ops.gen_array_ops import quantize_and_dequantize
sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')

import os
import gezi
from gezi import logging

from absl import app, flags
FLAGS = flags.FLAGS

import copy
import wandb
import json
import numpy as np
import pandas as pd
import pymp
from multiprocessing import Pool, Manager, cpu_count
import pymp
import collections
from collections import OrderedDict, Counter, defaultdict

from numba import njit
from scipy.stats import rankdata
import glob
import functools

from zipfile import ZIP_DEFLATED, ZipFile

import scipy
from sklearn.metrics.pairwise import cosine_similarity
import sklearn.preprocessing as skp
from sklearn import metrics
import sklearn.cluster as skc
from gezi import logging, tqdm

from qqbrowser import config
from qqbrowser.config import *
from qqbrowser.visualize import ImageLogger
from qqbrowser.util import *

queries, candidates, relevances, similarities = [], [], [], []

def clustering(m, n_clusters=100):
  if not n_clusters or n_clusters < 2:
    return m

  vid_embeddings = np.asarray(list(m.values()))
  cluster = skc.KMeans(n_clusters=n_clusters, random_state=0).fit(vid_embeddings)
  from sklearn.mixture import GaussianMixture
  clusters = cluster.predict(vid_embeddings)
  cluster_centers = cluster.cluster_centers_

  res = {}
  for i, key in enumerate(m):
    res[key] = cluster_centers[clusters[i]]
  
  return res

def eval_files(files, thre=None, normalize=False, n_clusters=None, weights=[], calc_sim=False):
  assert files
  vid_embeddings = []
  for file in tqdm(files):
    with open(file) as f:
      vid_embedding = json.load(f)
      vid_embeddings.append(vid_embedding)

  m = {}
  for key in vid_embedding:
    m[key] = []
  keys, l = [], []

  if calc_sim:
    m2 = {}

  for i, vid_embedding in enumerate(vid_embeddings):
    if calc_sim and i > 0:
      sim = 0.
      sims = [0.] * i
    for key in vid_embedding:
      if normalize:
        vid_embedding[key] = skp.normalize([vid_embedding[key]])[0]
      weight = 1. if not weights else weights[i]
      # assert weight == 1.

      if calc_sim and i > 0:
        sim += cosine_similarity([np.mean(np.asarray(m[key]), axis=0)], [vid_embedding[key]])[0][0]
        for j in range(i):
          sims[j] += cosine_similarity([np.asarray(m[key][j])], [vid_embedding[key]])[0][0]

      # m[key].append(vid_embedding[key])
      m[key].append(np.asarray([x * weight for x in vid_embedding[key]]))
    
    if calc_sim and i > 0:
      sim /= len(vid_embedding)
      for j in range(i):
        sims[j] /= len(vid_embedding)
      if i > 0:
        ic(i, files[i], f'{sim:.4f}', [f'{x:.4f}' for x in sims])

  ic(sum(weights), weights)
  if not weights:
    weights = [1.] * len(files)

  for key in vid_embedding:
    m[key] = list(np.sum(np.asarray(m[key]), axis=0) / sum(weights))
    if thre:
      m[key] = list(skp.normalize([np.asarray(m[key])])[0])
      l.append(m[key])
      keys.append(key)

  keys_dict = dict(zip(keys, range(len(keys))))

  # n_clusters = n_clusters or FLAGS.num_clusters
  # m = clustering(m, n_clusters)

  queries, candidates, relevances = [], [], []
  annotation_file = f'../input/pairwise/label_valid{FLAGS.fold_}.csv'
  for row in pd.read_csv(annotation_file).itertuples():
    query, candidate, relevance = str(row.query), str(row.candidate), row.relevance
    queries.append(query)
    candidates.append(candidate)
    relevances.append(float(relevance))

  if thre:
    emb_matrix = np.asarray(l)
    embs_sim = np.matmul(emb_matrix, np.transpose(emb_matrix))
    ic(embs_sim.shape)
    for i in range(len(embs_sim)):
      embs_sim[i][i] = -1e10
    ic(np.max(embs_sim))
    for i in range(len(embs_sim)):
      embs_sim[i][i] = 1e10
    ic(np.min(embs_sim))

    uset = gezi.UnionSet()
    m2 = {}
    for key in keys:
      m2[key] = copy.copy(m[key])
    for i, key in tqdm(enumerate(keys), total=len(keys), desc='merge'):
      for j, key2 in enumerate(keys):
        # if i == j:
        #   uset.join(i, i)
        if j > i:
          sim = embs_sim[i][j]
          if sim > thre:
            uset.join(i, j)
            embs_sim[i][j] = 1.
    ic(len(uset.clusters()))

    for i, cluster in enumerate(uset.clusters().values()):
      emb = [0.] * 256
      emb[i] = 1.
      # l = []
      # for idx in cluster:
      #   l.append(m2[keys[idx]])
      # emb = list(skp.normalize([list(np.mean(np.asarray(l), axis=0))])[0])
      # # ic(emb, cosine_similarity([emb], [emb])[0][0])
      # emb = tf.quantization.quantize(emb, -1., 1., tf.qint8).output.numpy()
      # emb = np.asarray(emb, dtype=np.float16)

      # ic(emb, cosine_similarity([emb], [emb])[0][0])
      for idx in cluster:
        m2[keys[idx]] = list(emb)
    # for key in m2:
    #   m2[key] = quantize(m2[key])

    m = m2

  same_label_count = 0
  same_pred_count = 0
  same_pred_count2 = 0
  relevances_, similarities = [], []
  for query, candidate, relevance in tqdm(zip(queries, candidates, relevances), total=len(queries), leave=False, desc='cosine'):
    if query in m and candidate in m:
      query_embedding = m[query]
      candidate_embedding = m[candidate]
      similarity = cosine_similarity([query_embedding], [candidate_embedding])[0][0]
      similarities.append(similarity)
      relevances_.append(relevance)
      if similarity == 1:
        same_pred_count += 1
      if abs(similarity - 1) < 1e-7:
        same_pred_count2 += 1
      if relevance == 1:
        same_label_count += 1
      # if thre:
      #   ic(query, candidate, relevance, similarity, 
      #      relevance == similarity,
      #      embs_sim[keys_dict[query]][keys_dict[candidate]], 
      #      np.dot(np.asarray(query_embedding), np.asarray(candidate_embedding)),
      #      np.sum(np.asarray(query_embedding) * np.asarray(query_embedding)),
      #      np.sum(np.asarray(candidate_embedding) * np.asarray(candidate_embedding)))
  
  ic(same_pred_count, same_pred_count2, same_label_count, same_label_count / len(query))

  similarities_ = np.asarray(similarities)  
  relevances_ = np.asarray(relevances_)
  spearmanr = scipy.stats.spearmanr(similarities_, relevances_[:len(similarities_)]).correlation

  ic(spearmanr)

  return spearmanr

def evaluate(y_true, y_pred, x, others, step, is_last=True):       
  # TODO precision recall for tag prediction?
  global queries, candidates, relevances, similarities
  if is_pointwise(): 
    m = dict(zip(x['vid'], others['final_embedding']))
    
    if not relevances:
      annotation_file = '../input/pairwise/label.tsv'
      with open(annotation_file, 'r') as f:
        for line in f:
          query, candidate, relevance = line.split()
          queries.append(query)
          candidates.append(candidate)
          relevances.append(float(relevance))

    similarities.clear()
    for query, candidate in tqdm(zip(queries, candidates), total=len(queries), leave=False, desc='cosine'):
      query_embedding = m[query]
      candidate_embedding = m[candidate]
      similarity = cosine_similarity([query_embedding], [candidate_embedding])[0][0]
      similarities.append(similarity)

    similarities_ = np.asarray(similarities)  
    relevances_ = np.asarray(relevances)
  else:
    similarities_, relevances_ = y_pred, y_true
    queries, candidates = x['vid1'], x['vid2']
    
  res = OrderedDict()
  res2 = OrderedDict()
  res2['true/len'] = len(relevances_)
  res2['true/mean'] = np.mean(relevances_)
  res2['pred/len'] = len(similarities_)
  res2['pred/mean'] = np.mean(similarities_)
  res2['pred/max'] = np.max(similarities_)
  res2['pred/min'] = np.min(similarities_)
  
  spearmanr = scipy.stats.spearmanr(similarities_, relevances_[:len(similarities_)]).correlation
  pearsonr = scipy.stats.pearsonr(similarities_, relevances_[:len(similarities_)])[0]
  auc = 1. - gezi.metrics.inverse_rate(relevances_[:len(similarities_)], similarities_)
  diff = similarities_ - relevances_[:len(similarities_)]
  l2_dist = np.mean(diff * diff)
  
  res['spearmanr'] = spearmanr
  res['pearsonr'] = pearsonr
  res['auc'] = auc
  res['l2'] = l2_dist
    
  top_tags, top_weights = None, None
  if 'top_tags' in others:
    top_tags = []
    top_weights = []
    
    if 'vid1' in x:
      m = {**dict(zip(x['vid1'], others['top_tags1'])), **dict(zip(x['vid2'], others['top_tags2']))}
      m2 = {**dict(zip(x['vid'], others['top_weights1'])), **dict(zip(x['vid2'], others['top_weights2']))}
    else:
      m = dict(zip(x['vid'], others['top_tags']))
      m2 = dict(zip(x['vid'], others['top_weights']))
    root = FLAGS.model_dir
    gezi.save_pickle(m, f'{root}/top_tags.pkl')
    gezi.save_pickle(m2, f'{root}/top_weights.pkl')
    for query, candidate in tqdm(zip(queries, candidates), total=len(queries), leave=False, desc='top_tags'):
      top_tags.append([m[query], m[candidate]])
      top_weights.append([m2[query], m2[candidate]])

    if is_pointwise():
      p3, p5, r1, r3, r5, r10 = [], [], [], [], [], []
      for vid, true_tags in zip(x['vid'], x['pos']):
        pred_tags = m[vid]
        true_tags = set([x for x in true_tags if x > 1])
        r = [int(x in true_tags) for x in pred_tags]
        p3.append(gezi.rank_metrics.precision_at_k(r, 3))
        p5.append(gezi.rank_metrics.precision_at_k(r, 5))
        r1.append(gezi.rank_metrics.recall_at_k(r, 1))
        r3.append(gezi.rank_metrics.recall_at_k(r, 3))
        r5.append(gezi.rank_metrics.recall_at_k(r, 5))
        r10.append(gezi.rank_metrics.recall_at_k(r, 10))
      res['p@3'] = np.mean(p3) 
      res['p@5'] = np.mean(p5)
      res['r@1'] = np.mean(r1) 
      res['r@3'] = np.mean(r3)
      res['r@5'] = np.mean(r10)

      # if (not FLAGS.remove_pred) and 'tags' in FLAGS.label_strategy:
      #   with gezi.Timer('tag metrics', print_fn=logging.debug):
      #     from_logits = False if FLAGS.final_activation else True
      #     if from_logits:
      #       y_prob = gezi.sigmoid(y_pred)
      #     else:
      #       y_prob = y_pred
      #     y_pred = (y_pred > 0.5).astype(np.int)
      #     p = np.random.permutation(len(y_true))
      #     # 2000 30s 10000 154s
      #     max_examples = FLAGS.max_eval_examples
      #     p = p[:max_examples]
      #     y_true, y_prob, y_pred = y_true[p], y_prob[p], y_pred[p]
      #     ranking_score = 1. - metrics.label_ranking_loss(y_true, y_prob)
      #     micro_score = metrics.average_precision_score(y_true, y_prob, average='micro')
      #     ## always none
      #     # macro_score = metrics.average_precision_score(y_true, y_prob, average='macro')  
          
      #     micro_precision = metrics.precision_score(y_true, y_pred, average='micro')
      #     macro_precision = metrics.precision_score(y_true, y_pred, average='macro')
          
      #     micro_recall = metrics.recall_score(y_true, y_pred, average='micro')
      #     macro_recall = metrics.recall_score(y_true, y_pred, average='macro')
          
      #     micro_f1 = metrics.f1_score(y_true, y_pred, average='micro')
      #     macro_f1 = metrics.f1_score(y_true, y_pred, average='macro')
      
      #     res['ranking_score'] = ranking_score
      #     res['micro_score'] = micro_score
      #     # res['macro_score'] = macro_score
          
      #     res['macro_precision']= macro_precision
      #     res['macro_recall']= macro_recall
      #     res['macro_f1'] = macro_f1
          
      #     res['micro_precision']= micro_precision
      #     res['micro_recall'] = micro_recall
      #     res['micro_f1'] = micro_f1

  if is_last and FLAGS.log_image:
    gezi.pprint(res, format='%.4f')
    ImageLogger(step, FLAGS.num_random, FLAGS.num_worst, 
                FLAGS.num_best, seed=FLAGS.vis_seed) \
        .log(queries, candidates, 
             relevances_, similarities_,
             top_tags, top_weights)
  
  if is_pointwise():
    res = gezi.dict_prefix(res, f'pointwise/')
  res = gezi.dict_prefix(res, 'Metrics/')
  res2 = gezi.dict_prefix(res2, 'Infos/')
  res = {**res, **res2}
  
  return res

def valid_write(ids, label, predicts, ofile, others={}):
  write_result(ids, predicts, ofile, others, is_infer=False)

def infer_write(ids, predicts, ofile, others={}):
  write_result(ids, predicts, ofile, others, is_infer=True)

def write_result(ids, predicts, ofile, others, is_infer=True):
  x = ids
  root = FLAGS.model_dir

  vid_embedding = {}
  if is_pointwise() or is_infer:
    vids = x['vid']
    embeddings = others['final_embedding'].astype(np.float16)
    for vid , embedding in zip(vids, embeddings):
      vid_embedding[vid] = embedding.tolist()
  else:
    vids = x['vid1']
    embeddings = others['final_embedding1'].astype(np.float16)
    for vid , embedding in zip(vids, embeddings):
      vid_embedding[vid] = embedding.tolist()
    vids = x['vid2']
    embeddings = others['final_embedding2'].astype(np.float16)
    for vid , embedding in zip(vids, embeddings):
      vid_embedding[vid] = embedding.tolist()
  
  out_json_file = 'result.json' if is_infer else ('valid.json' if not FLAGS.swap_train_valid else 'train.json')
  
  out_json = f'{root}/{out_json_file}'
  with open(out_json, 'w') as f:
    json.dump(vid_embedding, f)

  if is_infer:
    with ZipFile(f'{root}/result.zip', 'w', compression=ZIP_DEFLATED) as zip_file:
      zip_file.write(out_json, out_json_file)
  else:
    if is_pointwise():
      # ic(queries, candidates, relevances, similarities)
      df = pd.DataFrame({
        'query': queries,
        'candidate': candidates,
        'relevance': relevances, 
        'similarity': similarities
      })
      ic(len(df))
      df.to_csv(f'{root}/valid.csv', index=False)
    else:
      df = pd.DataFrame({
        'query': gezi.squeeze(x['vid1']),
        'candidate': gezi.squeeze(x['vid2']),
        'relevance': gezi.squeeze(x['relevance']), 
        'similarity': gezi.squeeze(predicts)
      })
      ic(len(df))
      df.to_csv(f'{root}/valid.csv', index=False)

def main(_):
  config.init()
  files = sys.argv[1:]
  ic(files)
  for file in files:
    ic(file, eval_files([file]))
  
  ic(eval_files(files, normalize=False))
  ic(eval_files(files, normalize=True))
  ic(eval_files(files, normalize=True, calc_sim=True))

if __name__ == '__main__':
  app.run(main)  