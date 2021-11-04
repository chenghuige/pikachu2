import melt.image.image_processing
from melt.image.image_processing import decode_image, read_image, convert_image, read_image_as, pred_augment, read_tiff_image
# from melt.image.image_processing import reate_image_model_init_fn  # not work now
# from melt.image.image_decoder import *
# from melt.image.image_processing import get_features_name, get_num_features, get_feature_dim, get_imagenet_from_checkpoint
# from melt.image.image_embedding import *
# from melt.image.image_model import *
try:
  from melt.image.augment import RandAugment, AutoAugment
except Exception as e:
  print(e)

try:
  import tensorflow as tf
  from classification_models.tfkeras import Classifiers
  import efficientnet.tfkeras as eff
  from gezi import logging
except Exception:
  pass

def get_classifier(name, is_official=False):
  if name.startswith('official_') or is_official:
    if name.startswith('official_'):
      name = name.split('_', 1)[-1]
    assert name[0].isupper(), name
    Model=getattr(tf.keras.applications, name)
  elif name.lower().startswith('eff'):
    assert name[0].isupper(), name
    Model = getattr(eff, name)
  else:
    try:
      Model, _ = Classifiers.get(name.lower())
    except Exception:
      assert name[0].isupper(), name
      Model = getattr(tf.keras.applications, name)
  return Model

def get_preprocessing(name, normalize_image=None):
  import segmentation_models as sm
  if name:
    if name.startswith('official_'):
      name = name.split('_', 1)[-1]
    if name.lower().startswith('efficient'):
      name = 'EfficientNetB0' 
    try:
      name_ = name.lower()
      preprocess = sm.get_preprocessing(name_)
      logging.info('using preprocess of sm for', name, name_)
      return preprocess
    except Exception as e:
      logging.warning('Not found preprocessing using segmentation_models for :', name, name_)
  if normalize_image == '0-1':
    preprocess = lambda x: tf.cast(x, tf.float32) / 255.
  elif normalize_image == '-1-1':
    preprocess = lambda x: tf.cast(x, tf.float32) / 127.5 - 1.
  else:
    preprocess = lambda x: x
  logging.info('using custom preprocess', preprocess, normalize_image)
  return preprocess

def img_sharpness(img):
  import numpy as np
  dx = np.diff(img)[1:,:] # remove the first row
  dy = np.diff(img, axis=0)[:,1:] # remove the first column
  dnorm = np.sqrt(dx**2 + dy**2)
  sharpness = np.average(dnorm)

def sharpness(filename):
  from PIL import Image
  import numpy as np

  im = Image.open(filename).convert('L') # to grayscale
  array = np.asarray(im, dtype=np.int32)

  gy, gx = np.gradient(array)
  gnorm = np.sqrt(gx**2 + gy**2)
  sharpness = np.average(gnorm)
  return sharpness

def resize(image, image_size, method='bilinear', pad=False):
  dtype = image.dtype
  if tf.__version__ < '2.4':
    # hack for fp16 train which is slow FIXED in tf 2.4 
    # https://github.com/tensorflow/tensorflow/issues/41934
    image = tf.cast(image, tf.float32)
  if not pad:
    image = tf.image.resize(image, image_size, method=method)
  else:
    image = tf.image.resize_with_crop_or_pad(image, image_size[0], image_size[1])
  # if tf.__version__ < '2.4':
  image = tf.cast(image, dtype)
  return image
