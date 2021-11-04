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
import numpy as np
from multiprocessing import Pool, Manager
from PIL import Image
from skimage import io

import melt
import gezi
from gezi import tqdm

# NUM_CLASSES = 8
NUM_CLASSES = 15 # compat with naic data version 2
imgs = None

# 0	建筑   2
# 1	耕地   3
# 2	林地   5
# 3	水体   0
# 4	道路   1
# 5	草地   4
# 6	其他   7
# 255	未标注区域  6 # will mask

m = {
  0: 2,
  1: 3,
  2: 5,
  3: 0,
  4: 1,
  5: 4,
  6: 7,
  255: 7,
}

def convert_mask(mask):
  f = np.vectorize(lambda x: m[x])
  mask = f(mask).astype(np.uint8)
  return mask

# 注意label文件是伪彩色png 将单一灰度值转换到rgb
def pil_imread(file_path):
  """read pseudo-color label"""
  im = Image.open(file_path)
  return np.asarray(im)

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
      feature['id'] = int(os.path.splitext(os.path.basename(img))[0][1:])
      feature['image'] = melt.read_image_as(img, 'png')
      mask = convert_mask(pil_imread(label).astype(np.uint8)) if label else ''
      feature['mask'] = melt.image.convert_image(Image.fromarray(mask), 'png') if mask != '' else ''
      if mask != '':
        feature['bins'] = np.bincount(mask.reshape(-1), minlength=NUM_CLASSES)
        feature['components'] = tf.reduce_max(tfa.image.connected_components(mask)).numpy()
      else:
        feature['bins'] = [0] * NUM_CLASSES
        feature['components'] = 0

      feature['is_train'] = int(FLAGS.mark == 'train')
      feature['src'] = 3
      
      writer.write_feature(feature)

def get_img_label_paths(images_path, labels_path):
  res = []
  for dir_entry in os.listdir(images_path):
    if os.path.isfile(os.path.join(images_path, dir_entry)):
      file_name, _ = os.path.splitext(dir_entry)
      res.append((os.path.join(images_path, file_name+".jpg"),
                  os.path.join(labels_path, file_name+".png")))
  return res

def main(data_dir):
  FLAGS.num_folds = FLAGS.num_folds_
  FLAGS.seed = FLAGS.seed_
  
  np.random.seed(FLAGS.seed)
  image_dir = f'{FLAGS.in_dir}/train_data/img_{FLAGS.mark}'
  label_dir = f'{FLAGS.in_dir}/train_data/lab_{FLAGS.mark}'
 
  print(image_dir, label_dir)
  global imgs
  if FLAGS.mark == 'train':
    imgs = get_img_label_paths(image_dir, label_dir)
    np.random.shuffle(imgs)
  else:
    imgs = glob.glob(f'{image_dir}/*.jpg')
  print(imgs[0], len(imgs))

  # build_features(30)
  with Pool(FLAGS.num_records) as p:
    p.map(build_features, range(FLAGS.num_records))

if __name__ == '__main__':
  flags.DEFINE_string('in_dir', '../input/baidu', '')
  flags.DEFINE_string('out_dir', '../input/baidu', '')
  flags.DEFINE_string('mark', 'train', 'train or test')
  flags.DEFINE_integer('num_records', 30, '')
  flags.DEFINE_integer('num_folds_', 10, '')
  flags.DEFINE_integer('seed_', 12345, '')
  flags.DEFINE_string('records_name', 'tfrecords', '')
  flags.DEFINE_bool('small', False, '')
  
  app.run(main) 
