"""
author: libowei
date:   2019.08.23
"""

import tensorflow as tf
import six
import os
import sys
import random
import time
import uuid


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


# codes copied from `text_dataset.py`
class Dataset():
    def __init__(self, feat_file_path, portrait_emb_dim):
        # self.Type = tf.data.TextLineDataset
        # self.batch_parse = FLAGS.batch_parse
        # self.index_addone = int(FLAGS.index_addone)
        # assert self.index_addone
        # self.max_feat_len = FLAGS.max_feat_len

        self.feat_file_path = feat_file_path
        self.portrait_emb_dim = portrait_emb_dim

        self.field_id = {}
        self.feat_to_field = {}
        self.feat_to_field_val = {}
        self.load_feature_files()
        # ---np.float32 much slower.. 1.0 -> 1.5h per epoch..
        self.float_fn = float

        # feature idx start from 4
        self.start = 4

    def load_feature_files(self):
        for line in open(self.feat_file_path, 'r'):
            if line == '':
                break
            line = line.rstrip()
            fields = line.split('\t')
            assert len(fields) == 2
            fid = int(fields[1])

            tokens = fields[0].split('\a')
            if tokens[0] not in self.field_id:
                self.field_id[tokens[0]] = len(self.field_id) + 1
            self.feat_to_field[fid] = self.field_id[tokens[0]]
            self.feat_to_field_val[fid] = tokens[1]
        # with open(FLAGS.field_file_path, 'w') as out:
        #     l = sorted(self.field_id.items(), key=lambda x: x[1])
        #     for filed, fid in l:
        #         print(filed, fid, sep='\t', file=out)

        # self.doc_emb_field_id = -1
        # self.user_emb_field_id = -1

        # if FLAGS.doc_emb_name in self.field_id:
        #   self.doc_emb_field_id = self.field_id[FLAGS.doc_emb_name]
        # if FLAGS.user_emb_name in self.field_id:
        #   self.user_emb_field_id = self.field_id[FLAGS.user_emb_name]

        # logging.info('----doc_emb_field_id', self.doc_emb_field_id)
        # logging.info('----user_emb_field_id', self.user_emb_field_id)

    def get_feat(self, fields):
        num_features = len(fields)
        feat_id = [None] * num_features
        feat_field = [None] * num_features
        feat_value = [None] * num_features

        for i in range(num_features):
            tokens = fields[i].split(':')
            feat_id[i] = int(tokens[0])
            feat_field[i] = self.feat_to_field[feat_id[i]]
            feat_value[i] = self.float_fn(tokens[1])

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
            cycle_profile_click = list(map(float, fields[-3].split(':')[1].split(',')))
            cycle_profile_show = list(map(float, fields[-2].split(':')[1].split(',')))
            cycle_profile_dur = list(map(float, fields[-1].split(':')[1].split(',')))
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
    for line in sys.stdin:
        # print("%s\t%s" % (random.randint(0, 31), line))  # add a column for shuffling
        print("%s\t%s" % (str(uuid.uuid4()), line))  # add a column for shuffling

# reducer: each reducer node generates one tfrecord file, then `hadoop fs -put` the file to HDFS.
# the last part of the tfrecord's filename  is the no. of samples.
def reducer(local_path, hdfs_path, feat_file_path, has_emb, use_emb, portrait_emb_dim):
    dataset = Dataset(feat_file_path, portrait_emb_dim)

    count = 0
    writer = tf.io.TFRecordWriter(local_path)
    for line in sys.stdin:

        fields = line.rstrip().split('\t')[1:]    # column 0 was added in mapper procedure
        if len(fields) > 4:
            click = int(fields[0])
            duration = int(fields[1])
            if duration > 60 * 60 * 12:
                duration = 60
            id = '{}\t{}'.format(fields[2], fields[3])
            uid = fields[2]
            if has_emb:
                feat_id, feat_field, feat_value, cycle_profile_click, cycle_profile_show, cycle_profile_dur = dataset.get_feat_portrait(fields[4:])
            else:
                feat_id, feat_field, feat_value = dataset.get_feat(fields[4:])
            feature = {
                'click': int64_feature(click),
                'duration': int64_feature(duration),
                'id': bytes_feature(id),
                'index': int64_feature(feat_id),
                'field': int64_feature(feat_field),
                'value': float_feature(feat_value),
                'user_click': int64_feature(0),  # use 0 temporarily
                'user_show': int64_feature(0),  # use 0 temporarily
                'user_duration': int64_feature(0)  # use 0 temporarily
            }

            if use_emb:
                feature['cycle_profile_click'] = float_feature(cycle_profile_click)
                feature['cycle_profile_show'] = float_feature(cycle_profile_show)
                feature['cycle_profile_dur'] = float_feature(cycle_profile_dur)

            example = tf.train.Example(features=tf.train.Features(feature=feature))

            writer.write(example.SerializeToString())
            count += 1

    writer.close()
    local_path_new = local_path + ".%s" % (count)
    os.system('mv ' + local_path + ' ' + local_path_new)
    os.system('hadoop fs -put ' + local_path_new + ' ' + hdfs_path)
    # os.system('rm -f ' + local_path_new)


if __name__ == '__main__':
    if sys.argv[1] == 'mapper':
        mapper()
    elif sys.argv[1] == 'reducer':
        local_path = './tfrecord.'+ str(random.randint(0, 1000000))
        hdfs_path = sys.argv[2]
        feat_file_path = sys.argv[3]
        has_emb = True if sys.argv[4]=="1" else False
        use_emb = True if sys.argv[5]=="1" else False
        portrait_emb_dim = int(sys.argv[6])
        reducer(local_path, hdfs_path, feat_file_path, has_emb, use_emb, portrait_emb_dim)
    else:
        print("bad para!")
        sys.exit()
