"""
author: libowei
date:   2019.08.23
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import six
import os
import sys
import random
import time
import uuid
import io
import json
try:
    import mmh3 
except Exception:
    pass

import numpy as np
import time
from datetime import datetime

# need this for config.py ..
sys.path.append('./')
from config import *

# codes copied from `melt`
def int64_feature(value):
    if not isinstance(value, (list, tuple)):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

# codes copied from `melt`
def bytes_feature(value):
    if not isinstance(value, (list, tuple)):
        value = [value]
    if not six.PY2:
        if isinstance(value[0], str):
            value = [x.encode() for x in value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


# codes copied from `melt`
def float_feature(value):
    if not isinstance(value, (list, tuple)):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def hash(x):
    return mmh3.hash64(x)[0] 

def get_time_interval(stamp, time_bins_per_hour=6):
  if not stamp:
    return 0
  x = time.localtime(stamp)
  span = int(60 / time_bins_per_hour)
  return x.tm_hour * time_bins_per_hour + int(x.tm_min / span) + 1

def get_weekday(stamp):
  if not stamp:
    return 0
  x = datetime.fromtimestamp(stamp)
  return x.weekday() + 1

# -- bowei
def get_timespan_interval(a, b):
    if not (a > 0 and b > 0 and a > b):
        return 0
    value = int(np.log2(a - b) * 5) + 1
    if value >200:
        return 200
    return value 


FEAT_START = NUM_PRES 
# codes copied from `text_dataset.py`
class Dataset():
    def __init__(self, field_file_path):
        self.field_file_path = field_file_path

        self.field_id = {}
 
        if self.field_file_path and self.field_file_path != 'none':
          self.load_feature_files()

        print('len field id', len(self.field_id), file=sys.stderr)

    def load_feature_files(self):
        for i, line in enumerate(open(self.field_file_path, 'r', encoding='utf-8')):
            if line == '':
                break
            line = line.rstrip()
            fields = line.split('\t')
            if len(fields) == 2:
              fid = int(fields[1])
            else:
              fid = i + 1

            self.field_id[fields[0]] = fid

        assert len(self.field_id) > 1 
        print('num fields', len(self.field_id), file=sys.stderr)

    def get_feat(self, fields):
        num_features = len(fields)
        feat_id = [None] * num_features
        feat_field = [None] * num_features
        feat_value = [None] * num_features

        for i in range(num_features):
            item = fields[i].split(":\b")
            feat = item[0]
            val = float(item[1]) if len(item) == 2 else 1.

            feat_id[i] = hash(feat.encode(ENCODING))
            if self.field_id and not HASH_ONLY:
                feat_field[i] = self.field_id[feat.split('\a')[0]]
            else:
                feat_field[i] = hash(feat.split('\a')[0].encode(ENCODING))
            feat_value[i] = val

            if not feat_id[i]:
                feat_id[i], feat_field[i], feat_value[i] = None, None, None

        feat_id = [x for x in feat_id if x is not None]
        feat_field = [x for x in feat_field if x is not None]
        feat_value = [x for x in feat_value if x is not None]

        return feat_id, feat_field, feat_value
    
    def get_tokens(self, field):
        maxlen = 100
        history_doc_ids = []
        keyword_ids = []
        topic_ids = []
        doc_id=[]
        doc_topic_id=[0]
        doc_kw_ids=[]

        for token in field.split("\b"):
            if token.startswith("doc_"):
                history_doc_ids.append(hash(token[len('doc_'):]))
            elif token.startswith("tp_"):
                topic_ids.append(hash(token[len('tp_'):].encode(ENCODING)))
            elif token.startswith("kw_"):
                keyword_ids.append(hash(token[len('kw_'):].encode(ENCODING)))
            elif token.startswith("art_kw_"):
                doc_kw_ids.append(hash(token[len('art_kw_'):].encode(ENCODING)))
            elif token.startswith("art_tp_"):
                doc_topic_id=[hash(token[len('art_tp_'):].encode(ENCODING))]

        history_doc_ids = history_doc_ids[:maxlen]
        keyword_ids = keyword_ids[:maxlen]
        topic_ids = topic_ids[:maxlen]
        doc_kw_ids = doc_kw_ids[:maxlen]

        return keyword_ids, topic_ids, history_doc_ids, doc_kw_ids, doc_topic_id


# mapper: add a uuid column for random partition, in order to shuffle data.
def mapper():
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='ignore')
    for line in sys.stdin:
        # print("%s\t%s" % (random.randint(0, 31), line))  # add a column for shuffling
        # add a column for random shuffling
        print("%s\t%s" % (str(uuid.uuid4()), line))
        # split by mid
        # print(line.split('\t', 1)[0], line, sep='\t')

# reducer: each reducer node generates one tfrecordss file, then `hadoop fs -put` the file to HDFS.
# the last part of the tfrecordss's filename  is the no. of samples.


def reducer(local_path, hdfs_path, field_file_path):
    dataset = Dataset(field_file_path)

    count = 0
    writer = tf.io.TFRecordWriter(local_path)
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='ignore')
    for i, line in enumerate(sys.stdin):
        if i == 0:
          print(i, line, file=sys.stderr)
        # column 0 was added in mapper procedure
        fields = line.rstrip().split('\t')[1:]
        fields_ = fields[:FEAT_START]
        if len(fields) > FEAT_START:
            click = int(fields[CLICK])
            duration = int(fields_[DUR])
            if duration > 60 * 60 * 12:
                duration = 60
            try:
                show_time = int(fields[SHOW_TIME])
            except Exception:
                show_time = 0
            unlike = int(fields[UNLIKE])
            num_interests = int(fields[ALL_INTEREST_CNT])
            id = '{}\t{}'.format(fields[MID], fields[DOCID])
            mid = fields[MID]
            docid = fields[DOCID]
            abtestid = int(fields[ABTESTID])
            position = int(fields[POSITION])
            type_ = int(MARK) # tuwen or video
            video_time = int(fields[VIDEO_TIME])
            product = fields[PRODUCT]
            try:
                impression_time = int(fields[IMPRESSION_TIME])
            except Exception:
                impression_time = 0

            feat_id, feat_field, feat_value = dataset.get_feat(fields[FEAT_START:])
            
            user_show = int(fields[USER_SHOW])
            ori_lr_score = float(fields[ORI_LR_SCORE])
            lr_score = float(fields[LR_SCORE])
            feature = {
                'click': int64_feature(click),
                'duration': int64_feature(duration),
                'ori_lr_score': float_feature(ori_lr_score),
                'lr_score': float_feature(lr_score),
                'abtestid': int64_feature(abtestid),
                'id': bytes_feature(id),
                'mid': bytes_feature(mid),
                'product': bytes_feature(product),
                'docid': bytes_feature(docid),
                'unlike': int64_feature(unlike),
                'num_interests': int64_feature(num_interests),
                'type': int64_feature(type_),
                'show_time': int64_feature(show_time),
                'video_time': int64_feature(video_time),
                'impression_time': int64_feature(impression_time),
                'position': int64_feature(position),
                'index': int64_feature(feat_id),
                'field': int64_feature(feat_field),
                'value': float_feature(feat_value),
                'user_show': int64_feature(user_show),
                'user_click': int64_feature(-1),  # use -1 temporarily
                'user_duration': int64_feature(-1)  # use -1 temporarily
            }

            keyword_ids, topic_ids, history_doc_ids, doc_kw_ids, doc_topic_id = dataset.get_tokens(fields[TOKENS])

            json_data = json.loads(fields[JSON_DATA], encoding='utf8')
            article_page_time = int(json_data[ARTICLE_PT])
            tw_history = list(map(hash,json_data[TW_HISTORY].split("\b")))
            vd_history = list(map(hash,json_data[VD_HISTORY].split("\b")))
            # device_info = list(map(hash,json_data[DEV_INFO]))
            rea = json_data.get(REA, "000")

            'time_interval', 'time_weekday', 'timespan_interval'

            time_interval = get_time_interval(impression_time) + 1
            time_weekday = get_weekday(impression_time) + 1
            timespan_interval = get_timespan_interval(impression_time, article_page_time) + 1

            feature.update({
                'keyword': int64_feature(keyword_ids),
                'topic': int64_feature(topic_ids),
                'history': int64_feature(history_doc_ids),
                'doc_keyword': int64_feature(doc_kw_ids),
                'doc_topic': int64_feature(doc_topic_id),
                'did': int64_feature([hash(docid)]),
                'uid': int64_feature([hash(mid)]),
                "tw_history": int64_feature(tw_history),
                "vd_history": int64_feature(vd_history),
                "article_page_time": int64_feature(article_page_time),
                'rea':bytes_feature(rea),
                'time_interval': int64_feature(time_interval),
                'time_weekday': int64_feature(time_weekday),
                'timespan_interval': int64_feature(timespan_interval)
                # 'device_info':int64_feature(device_info)
            })

            example = tf.train.Example(
                features=tf.train.Features(feature=feature))

            writer.write(example.SerializeToString())
            count += 1

    writer.close()
    
    local_path_new = local_path + ".%s" % (count)
    os.system('mv ' + local_path + ' ' + local_path_new)
    os.system('hadoop fs -put ' + local_path_new + ' ' + hdfs_path)
    # status = os.system('hadoop fs -test -e %s/%s' % (hdfs_path, local_path_new))
    # assert status == 0, '%d' % status
    # print(local_path, local_path_new, hdfs_path, status, file=sys.stderr)
    # os.system('rm -f ' + local_path_new)


if __name__ == '__main__':
    if sys.argv[1] == 'mapper':
        mapper()
    elif sys.argv[1] == 'reducer':
        local_path = './tfrecord.' + str(random.randint(0, 1000000))
        hdfs_path = sys.argv[2]
        field_file_path = sys.argv[3]
        print('------------field file path', field_file_path, file=sys.stderr)
        if not os.path.exists(field_file_path):
            field_file_path = None
  
        reducer(local_path, hdfs_path, field_file_path)
    else:
        print("bad para!")
        sys.exit()
