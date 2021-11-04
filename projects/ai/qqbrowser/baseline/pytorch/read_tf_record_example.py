import json

import numpy as np
from lichee.utils.tfrecord import example_pb2
from lichee.utils.tfrecord.reader import tfrecord_iterator

record_iterator = tfrecord_iterator('data/pairwise/pairwise.tfrecords')

with open('data/desc.json') as f:
    desc = json.load(f)

for i, record in enumerate(record_iterator):
    example = example_pb2.Example()
    example.ParseFromString(record)
    all_keys = list(example.features.feature.keys())
    print(all_keys)
    if i > 10:
        break
    for key in all_keys:
        if key == "frame_feature":  # skip frame feature
            continue
        type_name = desc[key]
        field = example.features.feature[key].ListFields()[0]
        value = field[1].value
        if type_name == "byte":
            value = bytes(np.frombuffer(value[0], dtype=np.uint8)).decode()
        elif type_name == "bytes":
            value = [bytes(np.frombuffer(v, dtype=np.uint8)).decode() for v in value]
        elif type_name == "float":
            value = np.array(value, dtype=np.float32)
        elif type_name == "int":
            value = np.array(value, dtype=np.int32)
        print(key, value)
