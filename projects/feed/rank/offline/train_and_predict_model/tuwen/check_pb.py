import tensorflow as tf
import sys

if len(sys.argv)<2:
    print("bad params!")
    exit(1)
model = sys.argv[1]
try:
    with tf.compat.v1.Session() as sess:
        with open(model, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
except:
    print("load pb file: %s error" % model)
    exit(1)
