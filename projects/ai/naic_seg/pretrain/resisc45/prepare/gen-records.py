from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app, flags
FLAGS = flags.FLAGS

import tensorflow as tf
import glob
import cv2
import numpy as np
from multiprocessing import Pool, Manager
from PIL import Image

import melt
import gezi
from gezi import tqdm

CLASSES = None
imgs = None

def build_features(index):
  global imgs
  fold = index % FLAGS.num_folds
  out_dir = f'{FLAGS.out_dir}/{FLAGS.records_name}/{fold}'
  gezi.try_mkdir(out_dir)
  ofile = f'{out_dir}/{index}.tfrec'

  print(ofile)
  with melt.tfrecords.Writer(ofile) as writer:
    num_imgs = len(imgs) if not FLAGS.small else 100
    for i in tqdm(range(num_imgs), ascii=True, desc=f'{index}_{fold}'):
      if i % FLAGS.num_records != index:
        continue

      img = imgs[i]

      feature = {}
      feature['id'] = os.path.splitext(os.path.basename(img))[0]
      if feature['id'].isdigit():
        feature['id'] = int(feature['id'])
      
      feature['image'] = melt.read_image(img)  
      # feature['image'] = melt.read_image_as(img, 'png')
      class_name = os.path.basename(os.path.dirname(img))
      feature['label'] = CLASSES[class_name]
      feature['cat'] = class_name
      writer.write_feature(feature)

def main(data_dir):
  FLAGS.num_folds = FLAGS.num_folds_
  FLAGS.seed = FLAGS.seed_
  
  np.random.seed(FLAGS.seed)
  image_dir = f'{FLAGS.in_dir}/NWPU-RESISC45'

  global CLASSES, imgs
  classes = glob.glob(f'{image_dir}/*')
  classes = [os.path.basename(x) for x in classes]
  print(classes)
  CLASSES = dict(zip(classes, range(len(classes))))
  print(CLASSES)

  imgs = glob.glob(f'{image_dir}/*/*.jpg')
  np.random.shuffle(imgs)
  print(imgs[0], len(imgs))

  # build_features(30)
  with Pool(FLAGS.num_records) as p:
    p.map(build_features, range(FLAGS.num_records))

if __name__ == '__main__':
  flags.DEFINE_string('in_dir', '../input', '')
  flags.DEFINE_string('out_dir', '../input', '')
  flags.DEFINE_string('mark', 'train', 'train or test')
  flags.DEFINE_integer('num_records', 40, '')
  flags.DEFINE_integer('num_folds_', 10, '')
  flags.DEFINE_integer('seed_', 12345, '')
  flags.DEFINE_string('records_name', 'tfrecords/NWPU-RESISC45', '')
  flags.DEFINE_bool('small', False, '')
  
  app.run(main) 
