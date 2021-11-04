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
import os
import gezi

import wandb
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, confusion_matrix
import collections
from collections import OrderedDict, Counter, defaultdict
from gezi import logging, tqdm
from .config import *

eval_data = None
evaluator = None

def prediction_to_ori(start_index, end_index, instance):
  if start_index > 0:
    orig_doc_start = instance['token_to_orig_map'][start_index]
    orig_doc_end = instance['token_to_orig_map'][end_index]

    final_answer = ""
    for idx in range(orig_doc_start, orig_doc_end + 1):
      if instance["prev_is_whitespace_flag"][idx] and idx > orig_doc_start:
        final_answer += " "

      final_answer += instance["doc_tokens"][idx]
    return final_answer
  else:
    logging.debug("pid: %d, start_index: %d, end_index: %d", instance["pid"],
                 start_index, end_index)
    return ""

def get_best_answer(output, other, instances, max_answer_len=100):
  def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits),
                             key=lambda x: x[1],
                             reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
      if i >= n_best_size:
        break
      best_indexes.append(index_and_score[i][0])
    return best_indexes

  pred_prob = {}
  na_prob = {}
  pred_text = {}
  pred_text2 = {}
  pred_prob2 = {}
  i = -1
  for instance in tqdm(instances, desc='get_best_answer', leave=False):
    if instance["qid"] == -1:
      continue

    i += 1

    if i == len(output):
      break

    start_logits = output[i, :, 0]
    end_logits = output[i, :, 1]
    start_indexes = _get_best_indexes(start_logits, n_best_size=20)
    end_indexes = _get_best_indexes(end_logits, n_best_size=20)

    max_start_index = -1
    max_end_index = -1
    max_start_index2 = -1
    max_end_index2 = -1
    max_logits = -100000000
    max_logits2 = -100000000
    for start_index in start_indexes:
      for end_index in end_indexes:
        if start_index >= len(instance['tokens']):
          continue
        if end_index >= len(instance['tokens']):
          continue
        if start_index not in instance['token_to_orig_map']:
          continue
        if end_index not in instance['token_to_orig_map']:
          continue
        if end_index < start_index:
          continue
        length = end_index - start_index - 1
        if length > max_answer_len:
          continue
        sum_logits = start_logits[start_index] + end_logits[end_index]
        if sum_logits > max_logits:
          max_start_index2 = max_start_index
          max_end_index2 = max_end_index
          max_logits2 = max_logits
          max_logits = sum_logits
          max_start_index = start_index
          max_end_index = end_index
        elif sum_logits > max_logits2:
          max_logits2 = sum_logits
          max_start_index2 = start_index
          max_end_index2 = end_index

    final_text = prediction_to_ori(max_start_index, max_end_index, instance)
    pred_text[instance["pid"]] = final_text
    pred_prob[instance["pid"]] = other["prob"][i][max_start_index][0] * other["prob"][i][max_end_index][1] 
    na_prob[instance["pid"]] = other["gate_prob"][i]
    # print('answer:', instance['answer'], 'final_text:', final_text, 'pred_prob:', pred_prob[instance["pid"]], 'na_prob:', na_prob[instance["pid"]])
    pred_text2[instance["pid"]] = prediction_to_ori(max_start_index2, max_end_index2, instance)
    pred_prob2[instance["pid"]] = other["prob"][i][max_start_index2][0] * other["prob"][i][max_end_index2][1]
  return pred_text, pred_prob, na_prob, pred_text2, pred_prob2

def remove_punctuation(in_str):
  in_str = str(in_str).lower().strip()
  sp_char = set(['-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=', \
             '，', '。', '：', '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、','、', \
             '「', '」', '（', '）', '－', '～', '『', '』'])
  out_segs = []
  for char in in_str:
    if char in sp_char:
      continue
    else:
      out_segs.append(char)
  return ''.join(out_segs)

def f1_score(pred, gts):
  max_f1 = 0.0
  for gt in gts:
    pred_toks = [c for c in remove_punctuation(pred).lower()]
    gold_toks = [c for c in remove_punctuation(gt).lower()]
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if num_same == 0:
      continue
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    max_f1 = max(max_f1, f1)

  return max_f1

def good_answer(pred, gts, relax=False):
  if f1_score(pred, gts) > 0.85:
    return True

  if relax:
    for gt in gts:
      if gt.lower() in pred.lower():
        return True

      if pred.lower() in gt.lower():
        return True

  return False

class NewDataGNEvaluator(object):
  def __init__(self, file_path):
    self.q_answer, self.qp_answer = dict(), dict()
    self.q_has_answer, self.qp_has_answer = dict(), dict()
    self.q2p, self.p2q = dict(), dict()

    self.p2question = {}
    self.p2passage = {}

    pid_list = list()

    num_lines = len([x for x in open(file_path)])
    with open(file_path) as dataset_file:
      for qp_line in tqdm(dataset_file, total=num_lines, leave=False):
        qp = json.loads(qp_line.strip("\n"))
        qid = qp["qid"]

        self.q_answer[qid] = set()
        self.q2p[qid] = list()

        for p in qp["passages"][:10]:
          pid = p["pid"]
          passage = p["passage"]
          answers = p["answer"]

          self.p2question[pid] = qp["question"]
          self.p2passage[pid] = p["passage"]

          self.qp_answer[(qid, pid)] = answers
          self.qp_has_answer[(qid, pid)] = (len(answers) > 0)
          pid_list.append(pid)

          if len(answers) > 0:
            self.q_answer[qid].add(answers)

          self.q2p[qid].append(pid)
          self.p2q[pid] = qid

        self.q_has_answer[qid] = (len(self.q_answer[qid]) > 0)

  def get_score(self,
                pred_info,
                p_low=0.9,
                p_high=1,
                p_interval=0.001,
                a_low=0.7,
                a_high=1,
                a_interval=0.001):
    pred_text, pred_prob, na_prob, pred_text2, pred_prob2 = pred_info

    em = dict()
    em_sum, cnt = 0.0, 0

    with open(f'{FLAGS.model_dir}/pred.txt', 'w') as f:
      for qp_id, has_answer in self.qp_has_answer.items():
        qid, pid = qp_id
        if has_answer:
          if pid not in pred_text:
            logging.debug('error', pid)
            continue
          em[pid] = int(good_answer(pred_text[pid],
                                    [self.qp_answer[(qid, pid)]]))
          print(qid, em[pid], 'pred:', pred_text[pid], 'pred2:', pred_text2[pid], 'gt:', self.qp_answer[(qid, pid)], 'na_prob:', na_prob[pid], 'pred_prob:', pred_prob[pid], 'pred_prob2:', pred_prob2[pid], 'ques:', self.p2question[pid], 'passage:', self.p2passage[pid], file=f)
          em_sum += em[pid]
          cnt += 1

    logging.debug("Passage-level has_answer EM: %f", em_sum / cnt)
    answer_em = em_sum / cnt

    find = False
    passage_recall = 0
    passage_recall_thre = 0.9
    if FLAGS.mode == 'valid':
      logging_ = logging.info
    else:
      logging_ = logging.debug
    # For passage-level precision recall
    logging_("Passage-level")
    for thres in tqdm(np.arange(p_low, p_high, p_interval), leave=False):
      precision, recall, pprecision, rrecall = 0, 0, 1e-8, 1e-8
      for pid, prob in na_prob.items():
        if prob > thres:
          pprecision += 1

          if self.qp_has_answer[(self.p2q[pid], pid)]:
            precision += 1

        if self.qp_has_answer[(self.p2q[pid], pid)]:
          rrecall += 1
          if prob > thres:
            recall += 1

      logging_("threshold: %f, precision: %f, recall: %f", thres,
                  float(precision) / pprecision,
                  float(recall) / rrecall)
      p_ = float(precision) / pprecision
      if not find and p_ > passage_recall_thre:
        passage_recall = float(recall) / rrecall
        find = True
        break

    # For query-level precision recall
    logging_("Query-level")

    find = False
    answer_recall = 0.
    # Second, get metric
    for thres in tqdm(np.arange(a_low, a_high, a_interval), leave=False):
      # First, aggregate answer for each query
      qid2ans = dict()
      # if qid == 'webqa_472bece461c83e1e3c08adc870d36fb6':
      #   print('---------------', thres)
      for qid, pid_list in self.q2p.items():
        ans2prob = defaultdict(float)
        for pid in pid_list:
          if pid not in na_prob or pid not in pred_text:
            if not FLAGS.num_valid:
              logging.debug('error1', pid)
            continue

          prob = na_prob[pid] * pred_prob[pid]
          prob2 = na_prob[pid] * pred_prob2[pid]
          # if qid == 'webqa_472bece461c83e1e3c08adc870d36fb6':
          #   print(pred_text[pid], prob, na_prob[pid], pred_prob[pid])
          #   print(pred_text2[pid], prob2, na_prob[pid], pred_prob2[pid])
          # if prob > thres and pred_text[pid] != "":
          #   ans2prob[pred_text[pid]] += prob
          # if prob2 > thres and pred_text[pid] != "":
          #   ans2prob[pred_text2[pid]] += prob2
          ans2prob[pred_text[pid]] += prob
          ans2prob[pred_text2[pid]] += prob2

        # print('before sort', ans2prob)
        ans2prob = sorted(ans2prob.items(), key=lambda x: x[1], reverse=True)
        # print('after sort', ans2prob)
        if len(ans2prob) > 0 and ans2prob[0][1] > thres:
          qid2ans[qid] = (ans2prob[0][0], ans2prob[0][1])
        else:
          qid2ans[qid] = ("", 0.0)

        # if qid == 'webqa_472bece461c83e1e3c08adc870d36fb6':
        #   print('----------------------------------')
        #   print(ans2prob)
        #   print(qid2ans[qid])

      answer_recall_thre = 0.85
      precision, recall, pprecision, rrecall = 0, 0, 1e-8, 1e-8
      for qid, ans_info in qid2ans.items():
        ans, prob = ans_info
        if ans != "":
          pprecision += 1
          if self.q_has_answer[qid] and f1_score(ans,
                                                self.q_answer[qid]) > 0.85:
            precision += 1
          elif thres > 0.663 and thres < 0.665:
            print('wrong:', qid, 'f1_score:', f1_score(ans, self.q_answer[qid]), 'ans:', ans, 'gt:', self.q_answer[qid])

        if self.q_has_answer[qid]:
          rrecall += 1
          if ans != "" and f1_score(ans, self.q_answer[qid]) > 0.85:
            recall += 1
          # elif thres > 0.9882 and thres < 0.9883:
          #   print('norecall:', qid, ans, f1_score(ans, self.q_answer[qid]), self.q_answer[qid])

      logging_("threshold: %f, precision: %f, recall: %f", thres,
                  float(precision) / pprecision,
                  float(recall) / rrecall)

      p_ = float(precision) / pprecision
      if not find and p_ > answer_recall_thre:
        answer_recall = float(recall) / rrecall
        find = True
        break

    passage_recall_thre = int(passage_recall_thre * 100)
    answer_recall_thre = int(answer_recall_thre * 100)
    return {
      'em': answer_em,
      f'passage_recall@P{passage_recall_thre}': passage_recall,
      f'answer_recall@P{answer_recall_thre}': answer_recall
    }

def evaluate(y_true, y_pred, x, other):
  global eval_data, evaluator
  if eval_data is None:
    eval_data = gezi.read_pickle(FLAGS.dev_pkl, use_timer=True)
    evaluator = NewDataGNEvaluator(FLAGS.dev_json)

  preds = get_best_answer(y_pred, other, eval_data)
  res = evaluator.get_score(preds)
  res = gezi.dict_prefix(res, 'Metrics/')
  return res

# --write_valid 控制写valid result, predicts对应model(input)的前向输出 other是一个dict 里面的key 对应model.out_keys
def valid_write(x, labels, predicts, ofile, other=None):
  total = len(labels)
  assert other
  x['pid'] = gezi.decode(x['pid'])
  x['qid'] = gezi.decode(x['qid'])
  with open(ofile, 'w') as f:
    print('qid,pid,na_prob,start_prob,end_prob', file=f)
    for i in tqdm(range(total), desc='valid_write', leave=False):  
      print(x['pid'][i], x['qid'][i], x['passage_has_answer'][i], other['gate_prob'][i], 
            other['prob'][i][x['start_position'][i]][0], 
            other['prob'][i][x['end_position'][i]][1], file=f, sep=',')
