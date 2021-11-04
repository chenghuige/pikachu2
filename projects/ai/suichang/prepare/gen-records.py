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
import cv2
from PIL import Image
import numpy as np
from multiprocessing import Pool, Manager

import melt
import gezi
from gezi import tqdm

imgs = None

# {
#   1: "耕地",
#   2: "林地",
#   3: "草地",
#   4: "道路",
#   5: "城镇建设用地",
#   6: "农村建设用地",
#   7: "工业用地",
#   8: "构筑物"
#   9: "水域"
#   10: "裸地"
#  }

CLASSES = ['farmland', 'forest', 'grass', 'road', 'urban_area', 'countryside', 'industrial_land', 'construction', 'water', 'bareland']
NUM_CLASSES = len(CLASSES)
WEIGHTS = [0.947, 0.998, 0.239, 0.847, 0.126, 0.5565, 0.083, 0.29887, 0.58169, 0.0426]
WEIGHTS = [1, 1, 1, 1, 2, 1, 3, 1, 1, 5]
WEIGHTS = np.asarray(WEIGHTS)

def build_features(index):
  global imgs
  fold = index % FLAGS.num_folds
  out_dir = f'{FLAGS.out_dir}/{FLAGS.records_name}/{FLAGS.mark}/{fold}'
  gezi.try_mkdir(out_dir)
  ofile = f'{out_dir}/{index}.tfrec'
  with melt.tfrecords.Writer(ofile, buffer_size=10000) as writer:
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
      feature['image'], feature['nir'] = melt.read_tiff_image(img, 'png')
      mask = ''
      if label:
        dtype = np.uint8
        mask = cv2.imread(label, cv2.IMREAD_UNCHANGED).astype(dtype)
        mask -= 1
      feature['mask'] = melt.image.convert_image(Image.fromarray(mask), 'png') if mask != '' else ''
      weight = 1
      if mask != '':
        bins = np.bincount(mask.reshape(-1), minlength=NUM_CLASSES).astype(np.bool).astype(np.int32)
        weight = np.max(WEIGHTS * bins)
      # if mask != '':
      #   feature['bins'] = np.bincount(mask.reshape(-1), minlength=NUM_CLASSES)
      #   feature['components'] = tf.reduce_max(tfa.image.connected_components(mask)).numpy()
      #   feature['classes'] = np.sum(feature['bins'] > 0).astype(np.int32)
      # else:
      #   feature['bins'] = [0] * NUM_CLASSES
      #   feature['components'] = 0
      #   feature['classes'] = 0

      feature['is_train'] = int(FLAGS.mark == 'train')
      feature['src'] = FLAGS.data_version

      # if feature['components'] < FLAGS.min_components:
      #   continue

      for _ in range(weight):
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

  FLAGS.num_folds = FLAGS.num_folds_
  FLAGS.seed = FLAGS.seed_

  if FLAGS.balance:
    FLAGS.records_name += '-balance'
  
  np.random.seed(FLAGS.seed)
  if FLAGS.mark == 'train':
    image_dir = f'{FLAGS.in_dir}/{FLAGS.mark}/image'
    label_dir = f'{FLAGS.in_dir}/{FLAGS.mark}/label'
  else:
    image_dir = f'{FLAGS.in_dir}/{FLAGS.mark}'
    label_dir = image_dir 
 
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
  flags.DEFINE_string('mark', 'train', 'train or test')
  flags.DEFINE_integer('num_records', 30, '6 gpu to infer')
  flags.DEFINE_integer('num_folds_', 10, '')
  flags.DEFINE_integer('seed_', 12345, '')
  flags.DEFINE_string('records_name', 'tfrecords', '')
  flags.DEFINE_bool('balance', False, '')
  flags.DEFINE_bool('small', False, '')
  flags.DEFINE_integer('data_version', 1, '1 初赛 2 复赛')
  flags.DEFINE_integer('min_components', 0, '')
  
  app.run(main) 
