from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')

from absl import app, flags
FLAGS = flags.FLAGS

import tensorflow as tf
import tensorflow_addons as tfa
import glob
import numpy as np
from multiprocessing import Pool, Manager, cpu_count
import pandas as pd
from collections import defaultdict
import pymp

import melt
import gezi
from gezi import tqdm
tqdm.pandas()

MAX_TAGS = 14
MAX_KEYS = 18
DESC_LEN = 128
DESC_CHAR_LEN = 256
WORD_LEN = 256
START_ID = 3
UNK_ID = 1
EMPTY_ID = 1
NAN_ID = -1
MAX_SHOWS = 50

df = None
user_vocab = None
doc_vocab = None
feeds = {}
history = {}
history_days = {}
feed_history_days = {}

doc_info = {}
doc_dynamic_feature = {}
default_vals = {}
todays = {}
todays2 = {}

KEYS = [
  'read_comment',
  'comment',
  'like',	
  'click_avatar',	
  'forward',	
  'follow',	
  'favorite',
  'play',	
  'stay',
]

ACTIONS = [
  'read_comment',
  'like',	
  'click_avatar',	
  'forward',
  'favorite',
  'comment',	
  'follow'
]

HIS_ACTIONS = None
DAYS = None

DOC_STATIC_FEATS = [
  # 'start_day',
]

SPANS = [1, 3, 7]

single_keys = ['author', 'song', 'singer']
multi_keys =  ['manual_tags', 'machine_tags', 'machine_tag_probs', 'machine_tags2', 'manual_keys', 'machine_keys', 'desc', 'desc_char', 'ocr', 'asr']
info_keys = single_keys + multi_keys
info_lens = [1] * len(single_keys) + [MAX_TAGS, MAX_TAGS, MAX_TAGS, MAX_TAGS, MAX_KEYS, MAX_KEYS, DESC_LEN, DESC_CHAR_LEN, WORD_LEN, WORD_LEN]

HIST_LENS = {
  "read_comment": 50,  # 是否查看评论
  "like": 50,  # 是否点赞
  "click_avatar": 30,  # 是否点击头像
  "forward": 20,  # 是否转发
  "favorite": 10,  # 是否收藏
  "comment": 3,  # 是否发表评论
  "follow": 5,  # 是否关注
  "pos": 50,
  "neg": 50,
  "finish": 50,
  "unfinish": 50,
  'stay': 50,  # TODO maybe remove from tfrecord
  'unstay': 20,
#   'unfinish2': 50,
#   'unstay2': 50,
#   "latest": 100, #  latest 包括不包括当天的show 对应所有都用户交互信息
#   "today": 100, # 当天的show
  # "show": 100, # 不包括当天的show 相当于 action | neg
}

# HIST_LENS = {
#   "read_comment": 200,  # 是否查看评论
#   "like": 100,  # 是否点赞
#   "click_avatar": 30,  # 是否点击头像
#   "forward": 20,  # 是否转发
#   "favorite": 10,  # 是否收藏
#   "comment": 10,  # 是否发表评论
#   "follow": 10,  # 是否关注
#   "pos": 50,
#   "neg": 50,
#   "finish": 50,
#   "unfinish": 50,
#   'stay': 50,
#   'unstay': 20,
# #   'unfinish2': 50,
# #   'unstay2': 50,
# #   "latest": 100, #  latest 包括不包括当天的show 对应所有都用户交互信息
# #   "today": 100, # 当天的show
#   # "show": 100, # 不包括当天的show 相当于 action | neg
# }

MAX_HIS_LEN = 1000

def cache_feed():
  df = pd.read_csv('../input/feed_info.csv')
  df = df.fillna(NAN_ID)
  for row in tqdm(df.itertuples(), total=len(df), ascii=False, desc='feed_info'):
    row = row._asdict()
    feeds[row['feedid']] = row

  try:
    df = pd.read_csv('../input/doc_static_feature.csv')
    for row in tqdm(df.itertuples(), total=len(df), ascii=False, desc='doc_static_feature'):
      row = row._asdict()
      feeds[row['feedid']].update(row)
  except Exception:
    pass
  
def is_neg(row):
  if 'actions' in row:
    return row['actions'] == 0
  else:
    return False
  
def get_history(userid, feedid, action, day, is_first):
  feeds, spans = history_days[day][userid][action]
  len_ = HIST_LENS[action] if not FLAGS.dynamic_len else MAX_HIS_LEN
  if not is_first:
    feeds_, spans_ = [], []
    for feed, span in zip(feeds, spans):
      if feed == feedid:
        continue
      feeds_.append(feed)
      spans_.append(span)
      if len(feeds_) == len_:
        return feeds_, spans_
    if not FLAGS.dynamic_len:
      feeds_, spans_ = gezi.pad(feeds_, len_), gezi.pad(spans_, len_)
    return feeds_, spans_
  
  if not FLAGS.dynamic_len:
    feeds, spans = gezi.pad(feeds, len_), gezi.pad(spans, len_)
  else:
    feeds, spans = feeds[:len_], spans[:len_]
  return feeds, spans

def get_feed_history(userid, feedid, action, day, is_first):
  if action in feed_history_days[day][feedid]:
    userids = feed_history_days[day][feedid][action] 
  else:
    return None
  len_ = HIST_LENS[action] if not FLAGS.dynamic_len else MAX_HIS_LEN
  if not is_first:
    userids_ = []
    for uid in userids:
      if uid == userid:
        continue
      userids_.append(uid)
      if len(userids_) == len_:
        return userids_
    if not FLAGS.dynamic_len:
      userids_ = gezi.pad(userids_, len_)
    return userids_
  
  if not FLAGS.dynamic_len:
    userids = gezi.pad(userids, len_)
  else:
    userids = userids[:len_]
  return userids

# TODO pre generate just load and search in tfrecord
def get_doc_dynamic_feats(fd, feedid, date, doc_dynamic_feats):
  key_ = f'{feedid}_{date}'
  if key_ in doc_dynamic_feats:
    return doc_dynamic_feats[key_]
  fe = {}
  fe['num_shows'] = sum(fd['shows'][:date])
  fe['num_shows2'] = sum(fd['shows'][:date + 1])
  fe['num_shows_today'] = fd['shows'][date]

  min_count = 10
  rate = min(fe['num_shows'] / min_count, 1.)
  for action in ['actions'] + ACTIONS:
    fe[f'num_{action}'] = sum(fd[action][:date])
    # TODO smooth ? for each action different default value from conf or FLAGS
    # 用均值取代
    # +0,+10 似乎并没有比 +1,+1效果好? 那么 0, 1?
    # a, b = 0, 10  rv=2
    a, b = 0, 1 #rv=3
    # a, b = 1, 1
    
    fe[f'{action}_rate'] = (fe[f'num_{action}'] + a) / (fe['num_shows'] + b) if rate else default_vals[action]
    # fe[f'{action}_rate'] = rate * ((fe[f'num_{action}']) / (fe['num_shows']) * (fe['num_shows'])) + (1 - rate) * default_vals[action] if rate else default_vals[action]

  for key in ['finish_rate', 'stay_rate']:
    fe[f'total_{key}'] = float(sum(fd[key][:date]))
    fe[f'{key}_mean'] = (fe[f'total_{key}'] + a) / (fe['num_shows'] + b) if rate else default_vals[key]
    # fe[f'{key}_mean'] = rate * (fe[f'total_{key}']) / (fe['num_shows'])  + (1 - rate) * default_vals[key] if rate else default_vals[key]

  for span in SPANS:
    fe[f'num_shows_{span}'] = sum(fd['shows'][date - span:date])
    rate = min(fe[f'num_shows_{span}'] / min_count, 1.)

    for action in ACTIONS + ['actions']:
      fe[f'num_{action}_{span}'] = sum(fd[action][date - span:date])
      # TODO smooth ?
      fe[f'{action}_rate_{span}'] = (fe[f'num_{action}_{span}'] + a) / (fe[f'num_shows_{span}'] + b)if rate else default_vals[action]

      # fe[f'{action}_rate_{span}'] = rate * (fe[f'num_{action}_{span}']) / (fe[f'num_shows_{span}']) + (1 - rate) * default_vals[action] if rate else default_vals[action]

    for key in ['finish_rate', 'stay_rate']:
      fe[f'total_{key}_{span}'] = float(sum(fd[key][date - span:date]))
      fe[f'{key}_mean_{span}'] = (fe[f'total_{key}_{span}'] + a) / (fe[f'num_shows_{span}'] + b) if rate else default_vals[key]
      # fe[f'{key}_mean_{span}'] = rate * (fe[f'total_{key}_{span}']) / (fe[f'num_shows_{span}']) + (1 - rate) * default_vals[key] if rate else default_vals[key]
    doc_dynamic_feats[key_] = fe
  return fe

def build_features(index):
  mark = FLAGS.day or FLAGS.mark
  if FLAGS.pred_file:
    mark = f'pred{mark}'
  out_dir = f'../input/{FLAGS.records_name}/{mark}'
  if FLAGS.neg_parts:
    out_dir += f'-{FLAGS.neg_parts}-{FLAGS.neg_part}'
  gezi.try_mkdir(out_dir)
  if index == 0:
    # notice ic() will cause tfrecord data loss when reading, might due to multiprocess conflict of icecream, to not use it here
    print(out_dir)
  ofile = f'{out_dir}/{index}.tfrec'

  total = len(df)
  start, end = gezi.get_fold(total, FLAGS.num_records, index)
  df_ = df.iloc[start:end]

  doc_dynamic_feats = {}

  # buffer_size = None if not 'train' in FLAGS.mark or FLAGS.sort_method else 10000
  buffer_size = FLAGS.buf_size # much faster with buffer_size so less write to disk
  with melt.tfrecords.Writer(ofile, buffer_size=buffer_size, shuffle=False) as writer:
    t = tqdm(df_.itertuples(), total=len(df_), desc=f'index:{index}', ascii=False)
    for row in t:
      # t.set_postfix({'index': index}) # this is slow..
      row = row._asdict()
      is_neg_row = is_neg(row)
      if FLAGS.neg_parts:
        if is_neg_row:
          rand_int = np.random.randint(FLAGS.neg_parts)
          if rand_int != FLAGS.neg_part:
            continue
      fe = {}
      fe['version'] = int(row['version']) if 'version' in 'row' else 2
      fe['userid'] = int(row['userid'])
      fe['feedid'] = int(row['feedid'])
      fe['doc'] = doc_vocab.id(fe['feedid'])
      fe['user'] = user_vocab.id(fe['userid'])
      assert(fe['doc'] > 1)
      assert(fe['user'] > 1)
      fe['date'] = int(row['date_']) if 'date_' in row else FLAGS.test_day
      fe['day'] = fe['date'] % 7
      fe['device'] = row['device']

      fe['finish_rate'] = row['finish_rate'] if 'finish_rate' in row else 1.
      fe['stay_rate'] = row['stay_rate'] if 'stay_rate' in row else 1.
      fe['is_first'] = row['is_first'] if 'is_first' in row else 1
      fe['is_neg'] = int(is_neg_row)
      fe['num_actions'] = row['actions'] if 'actions' in row else 0

      # ------feed静态特征
      feed = feeds[fe['feedid']]

      video_time = int(feed['videoplayseconds'])
      if video_time > 120:
        video_time = 62
      elif video_time > 60:
        video_time = 61
      fe['video_time'] = video_time + 1

      video_time2 = int(video_time / 10)
      if video_time > 120:
        video_time2 = 8
      elif video_time > 60:
        video_time2 = 7

      fe['video_time2'] = video_time2 + 1
      video_display = min(video_time, 60) / 60.
      fe['video_display'] = video_display

      for key in DOC_STATIC_FEATS:
        fe[key] = feed[key]

      fe['fresh'] = fe['date'] - feed['start_day'] if 'start_day' in feed else FLAGS.test_day

      # doc dynamic features
      if FLAGS.use_doc_dynamic:
        fd = doc_dynamic_feature[fe['feedid']]
        fe.update(get_doc_dynamic_feats(fd, fe['feedid'], fe['date'], doc_dynamic_feats))     

      # label
      for key in KEYS:
        fe[key] = float(row[key]) if key in row else 0.

      if FLAGS.use_history:
        for action in HIS_ACTIONS:
          # 注意per day的history_{day}.pkl是转换docid之后的
          fe[f'{action}s'], fe[f'{action}s_spans'] = get_history(fe['userid'], fe['doc'], action, fe['date'], bool(fe['is_first']))

      if FLAGS.use_feed_history:
        for action in HIS_ACTIONS:
          # 注意per day的history_{day}.pkl是转换docid之后的
          ret = get_feed_history(fe['user'], fe['feedid'], action, fe['date'], bool(fe['is_first']))
          if ret is not None:
            fe[f'u_{action}s'] = ret
        
      if FLAGS.use_today:
        fe['todays'] = gezi.pad([doc_vocab.id(x) for x in todays[fe['date']][fe['userid']] if x != fe['feedid']], MAX_SHOWS)
        if FLAGS.use_feed_history:
          fe['u_todays'] = gezi.pad([user_vocab.id(x) for x in todays2[fe['date']][fe['feedid']] if x != fe['userid']], MAX_SHOWS)
      
      if doc_info:
        for key in doc_info:
          fe[key] = list(doc_info[key][fe['doc']])

      writer.write_feature(fe)

def main(data_dir):
  global df, user_vocab, doc_vocab, history, history_days
  global doc_info, doc_dynamic_feature, default_vals, DAYS, HIS_ACTIONS
  FLAGS.version = FLAGS.version_
  np.random.seed(FLAGS.seed_)

  user_vocab = gezi.Vocab('../input/user_vocab.txt')
  doc_vocab = gezi.Vocab('../input/doc_vocab.txt')

  if FLAGS.write_docinfo:
    doc_lookup_file = '../input/doc_lookup.npy' if FLAGS.rare_unk else '../input/doc_ori_lookup.npy'
    ic(doc_lookup_file)
    doc_lookup_npy = np.load(doc_lookup_file)
    doc_info = gezi.split_feats(doc_lookup_npy, info_keys, info_lens)

  if FLAGS.mark != 'test':
    filename = 'user_action2.feather'
    if not os.path.exists(f'../input/{filename}'):
      filename = 'user_action2.csv'
    if FLAGS.day or FLAGS.mark == 'valid':
      day = FLAGS.day or 14
      filename_ = f'user_action2_{day}.feather'
      if os.path.exists(f'../input/{filename_}'):
        filename = filename_
  else:
    if not FLAGS.test_file:
      if os.path.exists('../input/test_b.csv'):
        filename = 'test_b.csv'
      else:
        filename = 'test_a.csv'
    else:
      filename = FLAGS.test_file
      if os.path.exists(filename):
        filename = os.path.basename(filename)
  ic(filename)
  cache_feed()
  with gezi.Timer(f'read {filename}', True):
    action_file = f'../input/{filename}'
    read_fn = pd.read_feather if action_file.endswith('.feather') else pd.read_csv
    df = read_fn(action_file)  
    df = gezi.reduce_mem_usage(df, fp16=True)

  if 'test' in filename:
    for key in KEYS:
      df[key] = 0.

  if FLAGS.pred_file:
    if FLAGS.day == 14:
      df2 = pd.read_csv(FLAGS.pred_file)
      df = pd.merge(df, df2, how='left', on=['userid', 'feedid'], suffixes=['_x', ''])
      # 只加入部分 对另外user做验证
      df = df[df.userid % 2 == 0]    
    elif FLAGS.day == 15:
      # TODO 初赛的test_a, test_b都打上伪标签.. 三份数据加入训练
      pass
    else:
      raise ValueError(FLAGS.day)

  if FLAGS.mark == 'valid':
    df = df[df.date_ == FLAGS.eval_day]
    #按天训练的valid也就是day 14 数据 要保留初赛version=1的数据
    if not (FLAGS.sort_method and ('day' in FLAGS.sort_method or 'date' in FLAGS.sort_method)):
      df = df[df.version == FLAGS.version]
  elif FLAGS.mark == 'test':
    df['date_'] = FLAGS.test_day
  else:
    if 'train' in FLAGS.mark and not FLAGS.use_v1:
      df = df[df.version == FLAGS.version]
    if FLAGS.mark == 'train':
      if FLAGS.day is None:
        df = df[df.date_ < FLAGS.eval_day]
      else:
        df = df[df.date_ == FLAGS.day]
      
  DAYS = set(df.date_)
  ic(DAYS)

  # default_vals = gezi.read_pickle('../input/action_default_vals.pkl')

  # if FLAGS.mark == 'test':
  #   assert len(df[df.is_first == 0]) == 0
    
  if FLAGS.mark != 'test':
    ic('non_first rate', len(df[df.is_first == 0]) / len(df))

  if FLAGS.use_history:
    for day in tqdm(DAYS, desc='read history_days'):
      history_file = f'../input/history_{day}.pkl'
      ic(day, history_file)
      history_days[day] = gezi.read_pickle(history_file)

    if not HIS_ACTIONS:
      HIS_ACTIONS = list(list(history_days[list(DAYS)[0]].values())[0].keys())
      HIS_ACTIONS = [x for x in HIS_ACTIONS if x not in FLAGS.excl_actions]
      ic('his_actions', HIS_ACTIONS)

  if FLAGS.use_feed_history:
    for day in tqdm(DAYS, desc='read feed history_days'):
      history_file = f'../input/feed_history_{day}.pkl'
      ic(day, history_file)
      feed_history_days[day] = gezi.read_pickle(history_file)

  if FLAGS.use_doc_dynamic:
    ic('load doc dynamic feature')
    doc_dynamic_feature_file = '../input/doc_dynamic_feature.pkl'
    doc_dynamic_feature = gezi.load_pickle(doc_dynamic_feature_file)

  if FLAGS.use_today:
    today_shows = df.groupby(['userid', 'date_'])['feedid'].progress_apply(list).reset_index(name='feedids')
    for day in DAYS:
      todays[day] = {}
    for row in tqdm(today_shows.itertuples()):
      todays[row.date_][row.userid] = row.feedids

    if FLAGS.use_feed_history:
      today_shows = df.groupby(['feedid', 'date_'])['userid'].progress_apply(list).reset_index(name='userids')
      for day in DAYS:
        todays2[day] = {}
      for row in tqdm(today_shows.itertuples()):
        todays2[row.date_][row.feedid] = row.userids

  if 'train' in FLAGS.mark:
    if FLAGS.shuffle_train:
      with gezi.Timer(f'----------shuffle df {FLAGS.seed_}', True):
        df = df.sample(frac=1, random_state=FLAGS.seed_)
    if FLAGS.sort_method:
      if 'user' in FLAGS.sort_method:
        df['userid_hash'] = df.userid.apply(lambda x: gezi.hash(str(x)) % 1000000)

      if ('day' in FLAGS.sort_method or 'date' in FLAGS.sort_method) and 'user' in FLAGS.sort_method:
        df = df.sort_values(['date_', 'userid_hash'], ascending=[True, True])
      elif 'day' in FLAGS.sort_method or 'date' in FLAGS.sort_method:
        df = df.sort_values(['date_'], ascending=True)
      elif 'user' in FLAGS.sort_method:
        df = df.sort_values(['userid_hash'], ascending=True)
    
  df = df.fillna(NAN_ID)

  if not FLAGS.num_records:
    FLAGS.num_records = cpu_count() - FLAGS.ignore_cpus

  # if FLAGS.mark == 'train':
  #   FLAGS.num_records *= 15
  # elif FLAGS.mark == 'train-all':
  #   FLAGS.num_records *= 20

  with gezi.Timer('build_feature'):
    if FLAGS.debug:
      FLAGS.records_name = 'tfrecords.debug'
      build_features(0)
    elif FLAGS.index is not None:
      build_features(FLAGS.index)
    else:
      nw = max(min(cpu_count() - FLAGS.ignore_cpus, FLAGS.num_records), 1)
      if nw == FLAGS.num_records:
        ic(FLAGS.num_records)
        with Pool(FLAGS.num_records) as p:
          p.map(build_features, range(FLAGS.num_records))
      else:
        #这里注意要有足够的num_records数目 使得一次进程处理的数据量足够小 运行时间不要太长 否则一次并行会hang 
        # 预计简单使用openmp效果一样
        FLAGS.num_records = -(-FLAGS.num_records // nw) * nw
        ic('cpu_count:', cpu_count(), 'num_records', FLAGS.num_records)
        l = np.array_split(range(FLAGS.num_records), int(FLAGS.num_records / nw))
        for i in tqdm(range(len(l))):
          with Pool(len(l[i])) as p:
            p.map(build_features, l[i])
#       else:
#         FLAGS.num_records = -(-FLAGS.num_records // nw) * nw
#         ic('cpu_count:', cpu_count(), 'num_records', FLAGS.num_records)
#         l = np.array_split(range(FLAGS.num_records), int(FLAGS.num_records / nw))
#   #     ic(l)
#         for i in tqdm(range(len(l))):
#           with pymp.Parallel(len(l[i])) as p:
#             for j in p.range(len(l[i])):
#               build_feautres(l[i][j])

if __name__ == '__main__':
  flags.DEFINE_string('mark', 'train', 'train or valid or test or train_all')
  flags.DEFINE_integer('num_records', None, '')
  flags.DEFINE_integer('seed_', 12345, '')
  flags.DEFINE_string('records_name', 'tfrecords', '')
  flags.DEFINE_bool('balance', False, '')
  flags.DEFINE_bool('small', False, '')
  flags.DEFINE_integer('min_components', 0, '')
  flags.DEFINE_integer('eval_day', 14, '')
  flags.DEFINE_integer('test_day', 15, '')
  flags.DEFINE_integer('day', None, '')

  flags.DEFINE_integer('index', None, '')
  flags.DEFINE_bool('limit_history', True, '')
  flags.DEFINE_integer('most_history', 50, '')

  flags.DEFINE_integer('neg_parts', 0, '')
  flags.DEFINE_integer('neg_part', 0, '')

  flags.DEFINE_bool('shuffle_train', True, '')

  flags.DEFINE_string('sort_method', None, '')
  flags.DEFINE_alias('sortby', 'sort_method')

  flags.DEFINE_bool('his_cls', False, '')
  flags.DEFINE_bool('exclude_nonfirst', False, '')

  flags.DEFINE_bool('write_docinfo', False, '')
  flags.DEFINE_bool('rare_unk', False, '')

  flags.DEFINE_string('test_file', None, '')

  flags.DEFINE_bool('compat', False, '')
  flags.DEFINE_bool('his_through', False, '')

  flags.DEFINE_bool('use_history', True, '')
  flags.DEFINE_bool('use_feed_history', True, '')
  flags.DEFINE_bool('use_doc_dynamic', False, '')
  flags.DEFINE_bool('dynamic_len', False, '')
  flags.DEFINE_bool('use_today', True, '是否记录当天展现')
  flags.DEFINE_bool('long_history', False, '')
  flags.DEFINE_bool('shuffle_history', False, '')

  flags.DEFINE_string('pred_file', None, '')

  flags.DEFINE_list('excl_actions', [], '')

  flags.DEFINE_bool('use_v1', True, '')
  flags.DEFINE_integer('version_', 2, '')
  flags.DEFINE_integer('ignore_cpus', 0, 'for v100 18 core will hang if not set 4 here..')

  flags.DEFINE_integer('buf_size', 10000, '')
  
  app.run(main) 
