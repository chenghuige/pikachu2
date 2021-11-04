import datasets
import fast_scnn
import deeplabv3plus
import bayes_segnet
import separable_unet
import unet
from tensorflow import keras
import tensorflow as tf
import cityscapesscripts.helpers.labels as labels
import numpy as np
import PIL
import argparse
import sys
import os

gpus = tf.config.list_physical_devices('GPU')
# Set Memory Growth to alleviate memory issues
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.set_visible_devices(gpus, 'GPU')

#----------IMPORT MODELS AND UTILITIES----------#

#----------CONSTANTS----------#
CITYSCAPES_LABELS = [
    label for label in labels.labels if -1 < label.trainId < 255]
# Add unlabeled
CITYSCAPES_LABELS.append(labels.labels[0])
# Get the IDS
CITYSCAPES_IDS = [label.id for label in CITYSCAPES_LABELS]
CLASSES = 20

#----------ARGUMENTS----------#
parser = argparse.ArgumentParser(
    prog='predict', description="Generate predictions from a batch of images from cityscapes")
parser.add_argument(
    "-m", "--model",
    help="Specify the model you wish to use: OPTIONS: unet, separable_unet, bayes_segnet, deeplabv3+, fastscnn",
    choices=['unet', 'bayes_segnet', 'deeplabv3+',
             'fastscnn', 'separable_unet'],
    required=True)
parser.add_argument(
    "-w", "--weights",
    help="Specify the weights path",
    type=str)
parser.add_argument(
    '-p', "--path",
    help="Specify the root folder for cityscapes dataset, if not used looks for CITYSCAPES_DATASET environment variable",
    type=str)
parser.add_argument(
    '-r', "--results-path",
    help="Specify the path for results",
    type=str)
parser.add_argument(
    '-c', "--coarse",
    help="Use the coarse images", action="store_true")
parser.add_argument(
    '-t', "--target-size",
    help="Set the image size for training, should be a elements of a tuple x,y,c",
    default=(1024, 2048, 3),
    type=tuple)
parser.add_argument(
    '--backbone',
    help="The backbone for the deeplabv3+ model",
    choices=['mobilenetv2', 'xception']
)

args = parser.parse_args()
# Get model_name
model_name = args.model
# Check the CITYSCAPES_ROOT path
if os.path.isdir(args.path):
    CITYSCAPES_ROOT = args.path
elif 'CITYSCAPES_DATASET' in os.environ:
    CITYSCAPES_ROOT = os.environ.get('CITYSCAPES_DATASET')
else:
    parser.error("ERROR: No valid path for Cityscapes Dataset given")
# Now do the target size
if args.target_size is None:
    target_size = (1024, 2048, 3)
else:
    target_size = args.target_size
# Check the path of weights
if not os.path.isfile(args.weights):
    parser.error("ERROR: Weights File not found.")
# Ensure results dir is made
if args.results_path is not None:
    if not os.path.isdir(args.results_path):
        os.makedirs(args.results_path)
else:
    parser.error("No results path specified")

files = datasets.get_cityscapes_files(
    CITYSCAPES_ROOT, 'leftImg8bit', 'val', 'leftImg8bit')

# Get the model
if model_name == 'unet':
    model = unet.model(input_size=target_size, num_classes=CLASSES)
if model_name == 'separable_unet':
    model = separable_unet.model(input_size=target_size, num_classes=CLASSES)
elif model_name == 'bayes_segnet':
    model = bayes_segnet.model(num_classes=CLASSES, input_size=target_size)
elif model_name == 'fastscnn':
    model = fast_scnn.model(input_size=target_size, num_classes=CLASSES)
elif model_name == 'deeplabv3+':
    if args.backbone is None:
        backbone = 'mobilenetv2'
    else:
        backbone = args.backbone
    model = deeplabv3plus.model(input_size=target_size, num_classes=CLASSES,
                                depthwise=True, backbone=backbone, output_stride=8)

# Load the weights
model.load_weights(args.weights)

for image_path in files:
    image_string = tf.io.read_file(image_path)
    image = tf.image.decode_png(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    if args.target_size != (1024, 2048, 3):
        image = tf.image.resize(
            image, (args.target_size[0], args.target_size[1]), method='nearest')
    image = tf.expand_dims(image, 0)
    prediction = model.predict(image)
    prediction = tf.reshape(tf.argmax(prediction, axis=-1),
                            (target_size[0], target_size[1], 1))
    prediction = np.matmul(tf.one_hot(prediction, CLASSES), CITYSCAPES_IDS)
    pil = tf.keras.preprocessing.image.array_to_img(
        prediction, data_format="channels_last", scale=False)
    pil.save(os.path.join(args.results_path, os.path.basename(image_path)))
