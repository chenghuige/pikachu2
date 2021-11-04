import cityscapesscripts.helpers.labels as labels
import tensorflow as tf
from tensorflow import keras
import numpy as np
import glob
import os
from functools import partial

CITYSCAPES_LABELS = [
    label for label in labels.labels if -1 < label.trainId < 255]
CITYSCAPES_COLORS = [label.color for label in CITYSCAPES_LABELS]
CITYSCAPES_IDS = [label.trainId for label in CITYSCAPES_LABELS]
NUM_THREADS = tf.data.experimental.AUTOTUNE


def get_cityscapes_files(root, type, split, image_type):
    """
    Utility function which simply gets all the names of the .png image
    files from a given root, type and split directories. 
    Designed to grab the cityscapes iles which have a given structure 
    once the zip files have been downloaded and extracted.

    The resulting list will be sorted by `sort()`

    :param root: (str) Root Directory for Cityscapes data
    :param type: (str) The type of Cityscapes Data we're dealing with (gtFine, leftImg8bit, gtCoarse)
    :param split: (str) The split which we're looking at, train, val or test data.
    :param image_type: (str) If leftImg8bit then this will be leftImg8bit, otherwise labelTrainIds

    :returns file_list: (list) A list of files (which are absolute paths)
    """
    file_list = glob.glob(os.path.join(
        root, type, split, '*/*'+image_type+'.png'))
    file_list.sort()
    return file_list


def augment(image, label, target_size=(1024, 1024, 3), classes=19):
    """
    The augmentation function which does the data augmentation, takes in a batch of
    images and labels and stacks them together.

    Once stacked, random flips and random crops will be performed on the stack.

    It will unstack them, Resize the images and labels to the target_size.

    The images will have a random brightness augmentation done on them.
    The values of those images will be clipped to ensure they are between 0 and 1.

    The labels are encoded using `tf.one_hot` to be in the same format as what the 
    models will output. This makes for easy use of the `categorical_crossentropy`
    loss function in keras.

    References:
    https://github.com/PacktPublishing/Hands-On-Computer-Vision-with-TensorFlow-2/blob/master/Chapter06/ch6_nb4_preparing_data_for_smart_car_apps.ipynb

    :param image: (Tensor) A batch of images, or a single image, assumed to be RGB.
    :param label: (Tensor) A batch of labels (or image masks), assumed to be grayscale

    :returns image_transformed: (Tensor) Return the images in shape of [target_size[0], target_size[1], 3]
    :returns label_transformed: (Tensor) Return one-hot encoded labels in shape of [target_size[0], target_size[1], classes]
    """
    # Get the original shapes
    original_shape = tf.shape(image)[-3:-1]
    num_image_channels = tf.shape(image)[-1]

    stacked_images = tf.concat(
        [image, tf.cast(label, dtype=image.dtype)], axis=-1)
    num_stacked_channels = tf.shape(stacked_images)[-1]

    # Randomly applied horizontal flip
    stacked_images = tf.image.random_flip_left_right(stacked_images)
    # Randomly applied vertical flip
    stacked_images = tf.image.random_flip_up_down(stacked_images)

    # Random cropping upto 40% of the image will be cropped
    random_scale_factor = tf.random.uniform(
        [], minval=.6, maxval=1., dtype=tf.float32)
    crop_shape = tf.cast(tf.cast(original_shape, tf.float32)
                         * random_scale_factor, tf.int32)

    # If its a single image
    if len(stacked_images.shape) == 3:
        # Do the crop shape as so
        crop_shape = tf.concat([crop_shape, [num_stacked_channels]], axis=0)
    # Otherwise its a batch
    else:
        # Get the batch size
        batch_size = tf.shape(stacked_images)[0]
        # Now adjust the crop size so its a 4D tensor
        crop_shape = tf.concat(
            [[batch_size], crop_shape, [num_stacked_channels]], axis=0)
    # Do the crop
    stacked_images = tf.image.random_crop(stacked_images, crop_shape)

    # Retreive the images
    image_transformed = stacked_images[..., :num_image_channels]
    # Resize to Target Size
    image_transformed = tf.image.resize(
        image_transformed, (target_size[0], target_size[1]), method='nearest')
    # Adjust brightness of image
    image_transformed = tf.image.random_brightness(
        image_transformed, max_delta=25/255)
    # Clip the values that are less than 0 or more than 1
    image_transformed = tf.clip_by_value(image_transformed, 0.0, 1.0)

    # Retrieve the labels
    label_transformed = tf.cast(
        stacked_images[..., num_image_channels:], dtype=label.dtype)
    # Resize the label
    label_transformed = tf.image.resize(
        label_transformed, (target_size[0], target_size[1]), method='nearest')
    # Clip the 255 values down to 19
    label_transformed = tf.clip_by_value(label_transformed, 0, 19)
    # One Hot encode the label
    label_transformed = tf.one_hot(label_transformed, classes, dtype=tf.int32)
    # Reshape it so its shape is [target_size[0], target_size[1], classes]
    label_transformed = tf.reshape(
        label_transformed, (target_size[0], target_size[1], classes))

    return image_transformed, label_transformed


def parse_function(image, label, target_size=(1024, 1024, 3), training=True, classes=19):
    """
    This will get the byte string, decode it, 
    scale them if necessary and return them
    resized as per necessary.

    References:
    http://cs230.stanford.edu/blog/datapipeline/

    :param image: (str) An absolute Path to an image
    :param label: (str) An absolute Path to a label (image mask)
    :param target_size: (tuple) The image size required for the model

    :returns image: (Tensor) An Image Tensor of shape [target_size[0], target_size[1], 3]
    :returns label: (Tensor) A Label Tensor of shape [target_size[0], target_size[1], classes]
    """
    # Get the byte strings for the files
    image_string = tf.io.read_file(image)
    label_string = tf.io.read_file(label)

    # Decode the byte strings as png images
    image = tf.image.decode_png(image_string, channels=3)
    label = tf.image.decode_png(label_string, channels=1)

    # Convert the image to float32 and scale all values between 0 and 1
    image = tf.image.convert_image_dtype(image, tf.float32)
    label = tf.cast(label, tf.int32)

    # If training, then do augmentation
    if training:
        image, label = augment(
            image, label, target_size=target_size, classes=classes)
    # Otherwise, its a test or validation set
    else:
        # Resize the image and label for the model
        image = tf.image.resize(
            image, (target_size[0], target_size[1]), method='nearest')
        label = tf.image.resize(
            label, (target_size[0], target_size[1]), method='nearest')
        # Clip the 255 values down to 19
        label = tf.clip_by_value(label, 0, 19)
        # One hot the label, ready to determine the loss
        label = tf.one_hot(label, classes, dtype=tf.int32)
        # Reshape it so its shape is [target_size[0], target_size[1], classes]
        label = tf.reshape(label, (target_size[0], target_size[1], classes))
    # Return the image and label as tensors
    return image, label


def create_dataset(cityscapes_root, batch_size, epochs, target_size, train=True, coarse=False, classes=19,
                   buffer_size=1000, options=None, shard_size=None, shard_rank=None):
    if train and not coarse:
        images = get_cityscapes_files(
            cityscapes_root, 'leftImg8bit', 'train', 'leftImg8bit')
        labels = get_cityscapes_files(
            cityscapes_root, 'gtFine', 'train', 'labelTrainIds')
    elif not train and not coarse:
        images = get_cityscapes_files(
            cityscapes_root, 'leftImg8bit', 'val', 'leftImg8bit')
        labels = get_cityscapes_files(
            cityscapes_root, 'gtFine', 'val', 'labelTrainIds')
    # TODO COARSE IMAGES DATASET
    # Training Dataset
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    # Shard if the shard_size has been specified (meaning we're doing multi-node multi-gpu training)
    if shard_size is not None:
        ds = ds.shard(num_shards=shard_size, index=shard_rank)
    # Shuffle the data
    ds = ds.shuffle(buffer_size)
    # Map the parsing function using partial to pass through arguments
    ds = ds.map(partial(parse_function, target_size=target_size,
                        training=train, classes=classes), num_parallel_calls=NUM_THREADS)
    # Set the batch size
    ds = ds.batch(batch_size)
    # Repeat the above for numbe of epochs
    ds = ds.repeat(epochs)
    # Get the first batch ready for training
    ds = ds.prefetch(1)
    # Set Dataset options
    if options is not None:
        ds = ds.with_options(options)
    return ds
