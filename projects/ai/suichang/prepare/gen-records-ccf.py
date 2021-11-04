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
NUM_CLASSES = 15 # compat with naic data version 2
imgs = None

# 植被（标记1）、道路（标记2）、建筑（标记3）、水体（标记4）以及其他(标记0)
m = {
  4: 0,
  2: 1, 
  3: 2,
  0: 7,
  1: 3
}

def convert_mask(mask):
  f = np.vectorize(lambda x: m[x])
  mask = f(mask).astype(np.uint8)
  return mask

def build_features(index):
  global imgs
  fold = index 
  out_dir = f'{FLAGS.out_dir}/{FLAGS.records_name}/{FLAGS.mark}/{fold}'
  gezi.try_mkdir(out_dir)
  # ofile = f'{out_dir}/{index}.tfrec'
  with melt.tfrecords.MultiWriter(out_dir, 1000) as writer:
    num_imgs = len(imgs) if not FLAGS.small else 100
    for i in tqdm(range(num_imgs), ascii=True, desc=f'{fold}'):
      if not os.path.basename(imgs[i][0]).startswith(str(fold + 1)):
        continue

      if FLAGS.mark == 'train':
        img, label = imgs[i]
      else:
        image, label = imgs[i], None
      
      feature = {}
      feature['id'] = os.path.splitext(os.path.basename(img))[0]
      if feature['id'].isdigit():
        feature['id'] = int(feature['id'])
      else:
        group, idx = feature['id'].split('_')
        feature['id'] = int(group) * 1000000 + int(idx)

      assert isinstance(feature['id'], int), feature['id']
      
      ## 输入是tiff转成png存储
      feature['image'] = melt.read_image(img)
      mask = convert_mask(cv2.imread(label, cv2.IMREAD_UNCHANGED)) if label else '' 
      feature['mask'] = melt.image.convert_image(Image.fromarray(mask), 'png') if mask != '' else ''
      if mask != '':
        feature['bins'] = np.bincount(mask.reshape(-1), minlength=NUM_CLASSES)
        feature['components'] = tf.reduce_max(tfa.image.connected_components(mask)).numpy()
      else:
        feature['bins'] = [0] * NUM_CLASSES
        feature['components'] = 0

      feature['is_train'] = FLAGS.mark == 'train'
      feature['src'] = 4

      writer.write_feature(feature)

def get_img_label_paths(images_path, labels_path):
  res = []
  for dir_entry in os.listdir(images_path):
    if os.path.isfile(os.path.join(images_path, dir_entry)):
      file_name, _ = os.path.splitext(dir_entry)
      res.append((os.path.join(images_path, file_name + ".png"),
                  os.path.join(labels_path, file_name + ".png")))
  return res

def main(data_dir):
  # assert FLAGS.num_classes
  # global NUM_CLASSES
  # NUM_CLASSES = FLAGS.num_classes

  FLAGS.num_folds = FLAGS.num_folds_
  FLAGS.seed = FLAGS.seed_
  
  np.random.seed(FLAGS.seed)
  image_dir = f'{FLAGS.in_dir}/{FLAGS.mark}/image'
  label_dir = f'{FLAGS.in_dir}/{FLAGS.mark}/label'
 
  print(image_dir, label_dir)
  global imgs

  imgs = get_img_label_paths(image_dir, label_dir)
  np.random.shuffle(imgs)

  print(imgs[0], len(imgs))

  if FLAGS.debug:
    build_features(0)
  else:
    with Pool(FLAGS.num_records) as p:
      p.map(build_features, range(FLAGS.num_records))

if __name__ == '__main__':
  flags.DEFINE_string('in_dir', '../input/ccf_remote_dataset', '')
  flags.DEFINE_string('out_dir', '../input/ccf_remote_dataset', '')
  flags.DEFINE_string('mark', 'train', 'train or test')
  # flags.DEFINE_integer('num_classes', 5, '')
  flags.DEFINE_integer('num_records', 5, '')
  flags.DEFINE_integer('num_folds_', 5, '')
  flags.DEFINE_integer('seed_', 12345, '')
  flags.DEFINE_string('records_name', 'tfrecords', '')
  flags.DEFINE_bool('small', False, '')
  
  app.run(main) 
