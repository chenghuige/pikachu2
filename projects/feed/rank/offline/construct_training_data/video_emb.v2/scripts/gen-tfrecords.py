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
try:
    import mmh3 
except Exception:
    pass

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
    # shoud be this but now online c++ code has problem not int64 TODO
    # return mmh3.hash64(x)[0] 
    return mmh3.hash64(x)[0] % FEAT_DIM

FEAT_START = NUM_PRES 
# codes copied from `text_dataset.py`
class Dataset():
    def __init__(self, feat_file_path, portrait_emb_dim):
        self.feat_file_path = feat_file_path
        self.portrait_emb_dim = portrait_emb_dim

        self.field_id = {}
        self.feat_to_field = {}
        self.feat_to_field_val = {}
        self.feat_id = {}

        if self.feat_file_path and self.feat_file_path != 'none':
          self.load_feature_files()

        print('len feat id', len(self.feat_id), file=sys.stderr)

    def load_feature_files(self):
        for i, line in enumerate(open(self.feat_file_path, 'r', encoding='utf-8')):
            if line == '':
                break
            line = line.rstrip()
            fields = line.split('\t')
            if len(fields) == 2:
              fid = int(fields[1])
            else:
              fid = i + 1

            tokens = fields[0].split('\a')
            if tokens[0] not in self.field_id:
                self.field_id[tokens[0]] = len(self.field_id) + 1
            self.feat_to_field[fid] = self.field_id[tokens[0]]
            self.feat_to_field_val[fid] = tokens[1]
            self.feat_id[fields[0]] = fid
            
        assert len(self.field_id) > 1 and len(self.feat_id) > 100


    def get_feat(self, fields):
        num_features = len(fields)
        feat_id = [None] * num_features
        feat_field = [None] * num_features
        feat_value = [None] * num_features

        for i in range(num_features):
            item = fields[i].split(":\b")
            feat = item[0]
            if self.feat_id and feat not in self.feat_id:
                # print('-----not in dict', feat, file=sys.stderr)
                continue
            val = float(item[1]) if len(item) == 2 else 1.
            if self.feat_id and not HASH_ONLY:
                feat_id[i] = self.feat_id[feat]
            else:
                feat_id[i] = hash(feat)
            if self.feat_to_field and not HASH_ONLY:
                feat_field[i] = self.feat_to_field[feat_id[i]]
            else:
                feat_field[i] = hash(feat.split('\a')[0])
            feat_value[i] = val

            if not feat_id[i]:
                feat_id[i], feat_field[i], feat_value[i] = None, None, None

        feat_id = [x for x in feat_id if x is not None]
        feat_field = [x for x in feat_field if x is not None]
        feat_value = [x for x in feat_value if x is not None]

        return feat_id, feat_field, feat_value

    def get_feat_portrait(self, fields):
        has_portrait = True
        if ',' in fields[-1]:
            num_features = len(fields) - 3
        else:
            has_portrait = False
            num_features = len(fields)
        feat_id = [None] * num_features
        feat_field = [None] * num_features
        feat_value = [None] * num_features

        for i in range(num_features):
            tokens = fields[i].split(':')
            feat_id[i] = int(tokens[0])
            feat_field[i] = self.feat_to_field[feat_id[i]]
            feat_value[i] = self.float_fn(tokens[1])

        if has_portrait:
            cycle_profile_click = list(
                map(float, fields[-3].split(':')[1].split(',')))
            cycle_profile_show = list(
                map(float, fields[-2].split(':')[1].split(',')))
            cycle_profile_dur = list(
                map(float, fields[-1].split(':')[1].split(',')))
        else:
            cycle_profile_click = [0.] * self.portrait_emb_dim
            cycle_profile_show = [0.] * self.portrait_emb_dim
            cycle_profile_dur = [0.] * self.portrait_emb_dim

        assert len(cycle_profile_click) == self.portrait_emb_dim, fields
        assert len(cycle_profile_show) == self.portrait_emb_dim, fields
        assert len(cycle_profile_dur) == self.portrait_emb_dim, fields

        return feat_id, feat_field, feat_value, cycle_profile_click, cycle_profile_show, cycle_profile_dur


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


def reducer(local_path, hdfs_path, feat_file_path, has_emb, use_emb, portrait_emb_dim):
    dataset = Dataset(feat_file_path, portrait_emb_dim)

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
            show_time = int(fields[SHOW_TIME])
            unlike = int(fields[UNLIKE])
            num_interests = int(fields[ALL_INTEREST_CNT])
            id = '{}\t{}'.format(fields[MID], fields[DOCID])
            mid = fields[MID]
            docid = fields[DOCID]
            abtestid = int(fields[ABTESTID])

            if has_emb:
                feat_id, feat_field, feat_value, cycle_profile_click, cycle_profile_show, cycle_profile_dur = dataset.get_feat_portrait(
                    fields[FEAT_START:])
            else:
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
                'docid': bytes_feature(docid),
                'unlike': int64_feature(unlike),
                'num_interests': int64_feature(num_interests),
                'show_time': int64_feature(show_time),
                'index': int64_feature(feat_id),
                'field': int64_feature(feat_field),
                'value': float_feature(feat_value),
                'user_show': int64_feature(user_show),
                'user_click': int64_feature(-1),  # use -1 temporarily  
                'user_duration': int64_feature(-1)  # use -1 temporarily
            }

            if use_emb:
                feature['cycle_profile_click'] = float_feature(
                    cycle_profile_click)
                feature['cycle_profile_show'] = float_feature(
                    cycle_profile_show)
                feature['cycle_profile_dur'] = float_feature(cycle_profile_dur)

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
        feat_file_path = sys.argv[3]
        print('------------feat file path', feat_file_path, file=sys.stderr)
        if not os.path.exists(feat_file_path):
            feat_file_path = None
        has_emb = True if sys.argv[4] == "1" else False
        use_emb = True if sys.argv[5] == "1" else False
        portrait_emb_dim = int(sys.argv[6])
        reducer(local_path, hdfs_path, feat_file_path,
                has_emb, use_emb, portrait_emb_dim)
    else:
        print("bad para!")
        sys.exit()
