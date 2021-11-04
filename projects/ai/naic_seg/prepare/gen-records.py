from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app, flags
FLAGS = flags.FLAGS

import tensorflow as tf
import tensorflow_addons as tfa
import glob
import cv2
from PIL import Image
import numpy as np
from multiprocessing import Pool, Manager

import melt
import gezi
from gezi import tqdm

# NUM_CLASSES = 8
NUM_CLASSES = 15 # force data_version 1 compat with 2

imgs = None

m = {}
for i in range(17):
  if i < 4:
    m[i + 1] = i
  else:
    m[i + 1] = i - 2

def convert_mask(mask, pred=None):
  if FLAGS.data_version == 2:
    f = np.vectorize(lambda x: m[x])
  else:
    if pred is None:
      f = np.vectorize(lambda x: int(mask / 100 - 1))
    else:
      f = np.vectorize(lambda x: v1_to_v2(x, pred))
  mask = f(mask).astype(np.uint8)
  return mask

def get_pred(id):
  pred_file = f'{FLAGS.distill_dir}/{id}.npy'
  pred = np.load(pred_file)
  return pred

# CLASS_LOOKUP_SOFT = [
#   [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #water
#   [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #track
#   [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #build 
#   [0, 0, 0, 0, 0, 0, 0.95, 0.05, 0, 0, 0, 0, 0, 0, 0], #arable
#   [0, 0, 0, 0, 0, 0, 0, 0, 0.84, 0.16, 0, 0, 0, 0, 0], #grass
#   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.65, 0.35, 0, 0, 0], #forest
#   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.8, 0.2, 0], #bare
#   [0, 0, 0, 0, 0.711, 0.036, 0, 0, 0, 0, 0, 0, 0, 0, 0.253], #other
#   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# ]

# LABLE_MASK = [
#   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], #water
#   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], #track_ROAD
#   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], #build 
#   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], #track_airport
#   [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0], #ohter_park
#   [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0], #other_playground
#   [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1], #arable_natural
#   [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1], #arable_greenhouse
#   [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1], #grass_natural
#   [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1], #grass_greenbelt
#   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1], #forest_planted
#   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1], #bare_natural
#   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1], #bare_planted
#   [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0], #other_other
# ]

def v1_to_v2(x, pred):
  x = int(x / 100 - 1)
  if x < 3:
    return x
  if x == 3:
    return 6 + int(pred[7] > pred[6])
  if x == 4:
    return 8 + int(pred[9] > pred[8])
  if x == 5:
    return 10 + int(pred[11] > pred[10])
  if x == 6:
    return 12 + int(pred[13] > pred[12])
  if x == 7:
    l = [4, 5, 14]
    vals = np.asarray([pred[4], pred[5], pred[14]])
    return l[vals.argmax()]

def build_features(index):
  global imgs
  fold = index % FLAGS.num_folds
  out_dir = f'{FLAGS.out_dir}/{FLAGS.records_name}/{FLAGS.mark}/{fold}'
  gezi.try_mkdir(out_dir)
  ofile = f'{out_dir}/{index}.tfrec'
  with melt.tfrecords.Writer(ofile) as writer:
    num_imgs = len(imgs) if not FLAGS.small else 100
    for i in tqdm(range(num_imgs), ascii=True, desc=f'{FLAGS.mark}_{index}_{fold}'):
      if i % FLAGS.num_records != index:
        continue

      if FLAGS.mark != 'train':
        img, label = imgs[i], None
      else:
        img, label = imgs[i]
      
      feature = {}
      id =  os.path.splitext(os.path.basename(img))[0]
      feature['id'] = id
      if feature['id'].isdigit():
        feature['id'] = int(feature['id'])
      ## 输入是tiff转成png存储
      # feature['image'] = melt.read_image(img)  
      feature['image'] = melt.read_image_as(img, 'png')
      dtype = np.uint16 if FLAGS.data_version == 1 else np.uint8
      pred = None 
      if FLAGS.distill_dir:
        pred = get_pred(id).reshape(-1).tobytes()
      if pred is not None and FLAGS.write_distill:
        feature['pred'] = pred
      mask = convert_mask(cv2.imread(label, cv2.IMREAD_UNCHANGED).astype(dtype), pred=pred) if label else ''
      if mask == '' and pred is not None:
        mask = pred.argmax(axis=-1).astype(dtype)
      feature['mask'] = melt.image.convert_image(Image.fromarray(mask), 'png') if mask != '' else ''
      if mask != '':
        feature['bins'] = np.bincount(mask.reshape(-1), minlength=NUM_CLASSES)
        feature['components'] = tf.reduce_max(tfa.image.connected_components(mask)).numpy()
        feature['classes'] = np.sum(feature['bins'] > 0).astype(np.int32)
      else:
        feature['bins'] = [0] * NUM_CLASSES
        feature['components'] = 0
        feature['classes'] = 0

      feature['is_train'] = int(FLAGS.mark == 'train')
      feature['src'] = FLAGS.data_version

      if feature['components'] < FLAGS.min_components:
        continue

      writer.write_feature(feature)

def get_img_label_paths(images_path, labels_path):
  res = []
  for dir_entry in os.listdir(images_path):
    if os.path.isfile(os.path.join(images_path, dir_entry)):
      file_name, _ = os.path.splitext(dir_entry)
      res.append((os.path.join(images_path, file_name + ".tif"),
                  os.path.join(labels_path, file_name + ".png")))
  return res

def main(data_dir):
  global NUM_CLASSES
  if FLAGS.data_version == 2:
    NUM_CLASSES = 15
    FLAGS.in_dir += '/quarter'
    FLAGS.out_dir += '/quarter'

  if FLAGS.distill_dir:
    if FLAGS.write_distill:
      FLAGS.records_name += '.soft'
    else:
      FLAGS.records_name += '.hard'

  FLAGS.num_folds = FLAGS.num_folds_
  FLAGS.seed = FLAGS.seed_
  
  np.random.seed(FLAGS.seed)
  image_dir = f'{FLAGS.in_dir}/{FLAGS.mark}/image'
  label_dir = f'{FLAGS.in_dir}/{FLAGS.mark}/label'
  if not os.path.exists(image_dir):
    image_dir = f'{FLAGS.in_dir}/{FLAGS.mark}/images'
    label_dir = f'{FLAGS.in_dir}/{FLAGS.mark}/labels'
 
  print(image_dir, label_dir)
  global imgs
  if FLAGS.mark == 'train':
    imgs = get_img_label_paths(image_dir, label_dir)
    np.random.shuffle(imgs)
  else:
    imgs = glob.glob(f'{image_dir}/*.tif')
  print(imgs[0], len(imgs))

  if FLAGS.debug:
    build_features(0)
  else:
    with Pool(FLAGS.num_records) as p:
      p.map(build_features, range(FLAGS.num_records))

if __name__ == '__main__':
  flags.DEFINE_string('in_dir', '../input', '')
  flags.DEFINE_string('out_dir', '../input', '')
  flags.DEFINE_string('distill_dir', None, '')
  flags.DEFINE_bool('write_distill', True, '')
  flags.DEFINE_string('mark', 'train', 'train or test')
  flags.DEFINE_integer('num_records', 30, '6 gpu to infer')
  flags.DEFINE_integer('num_folds_', 10, '')
  flags.DEFINE_integer('seed_', 12345, '')
  flags.DEFINE_string('records_name', 'tfrecords', '')
  flags.DEFINE_bool('small', False, '')
  flags.DEFINE_integer('data_version', 2, '1 初赛 2 复赛')
  flags.DEFINE_integer('min_components', 0, '')
  
  app.run(main) 
