# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:16:00 2019

@author: Attila Lengyel
"""

import numpy as np
import scipy
from scipy import linalg
from scipy import ndimage
from PIL import Image as pil_image
from PIL import ImageEnhance

def array_to_img(x, data_format='channels_last', scale=False, dtype='float32'):
    """Converts a 3D Numpy array to a PIL Image instance.

    # Arguments
        x: Input Numpy array.
        data_format: Image data format.
            either "channels_first" or "channels_last".
        dtype: Dtype to use.

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
        ValueError: if invalid `x` or `data_format` is passed.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    x = np.asarray(x, dtype=dtype)
    if x.ndim != 3:
        raise ValueError('Expected image array to have rank 3 (single image). '
                         'Got array with shape: %s' % (x.shape,))

    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Invalid data_format: %s' % data_format)

    # Original Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but target PIL image has format (width, height, channel)
    if data_format == 'channels_first':
        x = x.transpose(1, 2, 0)
    if x.shape[2] == 4:
        # RGBA
        return pil_image.fromarray(x.astype('uint8'), 'RGBA')
    elif x.shape[2] == 3:
        # RGB
        return pil_image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return pil_image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise ValueError('Unsupported channel number: %s' % (x.shape[2],))


def img_to_array(img, data_format='channels_last', dtype='float32'):
    """Converts a PIL Image instance to a Numpy array.

    # Arguments
        img: PIL Image instance.
        data_format: Image data format,
            either "channels_first" or "channels_last".
        dtype: Dtype to use for the returned array.

    # Returns
        A 3D Numpy array.

    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: %s' % data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=dtype)
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: %s' % (x.shape,))
    return x

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def random_rotation(x, rg, row_axis=1, col_axis=2, channel_axis=0,
                    fill_mode='nearest', cval=0.):
    """Performs a random rotation of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        rg: Rotation range, in degrees.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        Rotated Numpy image tensor.
    """
    theta = np.random.uniform(-rg, rg)
    x = apply_affine_transform(x, theta=theta, channel_axis=channel_axis,
                               fill_mode=fill_mode, cval=cval)
    return x


def random_shift(x, wrg, hrg, row_axis=1, col_axis=2, channel_axis=0,
                 fill_mode='nearest', cval=0.):
    """Performs a random spatial shift of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        wrg: Width shift range, as a float fraction of the width.
        hrg: Height shift range, as a float fraction of the height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        Shifted Numpy image tensor.
    """
    h, w = x.shape[row_axis], x.shape[col_axis]
    tx = np.random.uniform(-hrg, hrg) * h
    ty = np.random.uniform(-wrg, wrg) * w
    x = apply_affine_transform(x, tx=tx, ty=ty, channel_axis=channel_axis,
                               fill_mode=fill_mode, cval=cval)
    return x


def random_shear(x, intensity, row_axis=1, col_axis=2, channel_axis=0,
                 fill_mode='nearest', cval=0.):
    """Performs a random spatial shear of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        intensity: Transformation intensity in degrees.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        Sheared Numpy image tensor.
    """
    shear = np.random.uniform(-intensity, intensity)
    x = apply_affine_transform(x, shear=shear, channel_axis=channel_axis,
                               fill_mode=fill_mode, cval=cval)
    return x


def random_zoom(x, zoom_range, row_axis=1, col_axis=2, channel_axis=0,
                fill_mode='nearest', cval=0.):
    """Performs a random spatial zoom of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        zoom_range: Tuple of floats; zoom range for width and height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        Zoomed Numpy image tensor.

    # Raises
        ValueError: if `zoom_range` isn't a tuple.
    """
    if len(zoom_range) != 2:
        raise ValueError('`zoom_range` should be a tuple or list of two'
                         ' floats. Received: %s' % (zoom_range,))

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    x = apply_affine_transform(x, zx=zx, zy=zy, channel_axis=channel_axis,
                               fill_mode=fill_mode, cval=cval)
    return x


def apply_channel_shift(x, intensity, channel_axis=0):
    """Performs a channel shift.

    # Arguments
        x: Input tensor. Must be 3D.
        intensity: Transformation intensity.
        channel_axis: Index of axis for channels in the input tensor.

    # Returns
        Numpy image tensor.

    """
    x = np.rollaxis(x, channel_axis, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [
        np.clip(x_channel + intensity,
                min_x,
                max_x)
        for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def random_channel_shift(x, intensity_range, channel_axis=0):
    """Performs a random channel shift.

    # Arguments
        x: Input tensor. Must be 3D.
        intensity_range: Transformation intensity.
        channel_axis: Index of axis for channels in the input tensor.

    # Returns
        Numpy image tensor.
    """
    intensity = np.random.uniform(-intensity_range, intensity_range)
    return apply_channel_shift(x, intensity, channel_axis=channel_axis)


def apply_brightness_shift(x, brightness):
    """Performs a brightness shift.

    # Arguments
        x: Input tensor. Must be 3D.
        brightness: Float. The new brightness value.
        channel_axis: Index of axis for channels in the input tensor.

    # Returns
        Numpy image tensor.

    # Raises
        ValueError if `brightness_range` isn't a tuple.
    """
    if ImageEnhance is None:
        raise ImportError('Using brightness shifts requires PIL. '
                          'Install PIL or Pillow.')
    y = x[:,:,3:]
    x = array_to_img(x[:,:,:3])
    x = imgenhancer_Brightness = ImageEnhance.Brightness(x)
    x = imgenhancer_Brightness.enhance(brightness)
    x = img_to_array(x).astype('uint8')
    x = np.concatenate((x,y),axis=2)
    return x


def random_brightness(x, brightness_range):
    """Performs a random brightness shift.

    # Arguments
        x: Input tensor. Must be 3D.
        brightness_range: Tuple of floats; brightness range.
        channel_axis: Index of axis for channels in the input tensor.

    # Returns
        Numpy image tensor.

    # Raises
        ValueError if `brightness_range` isn't a tuple.
    """
    if len(brightness_range) != 2:
        raise ValueError(
            '`brightness_range should be tuple or list of two floats. '
            'Received: %s' % (brightness_range,))

    u = np.random.uniform(brightness_range[0], brightness_range[1])
    return apply_brightness_shift(x, u)


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def apply_affine_transform(x, theta=0, tx=0, ty=0, shear=0, zx=1, zy=1,
                           row_axis=0, col_axis=1, channel_axis=2,
                           fill_mode='nearest', cval=0.):
    """Applies an affine transformation specified by the parameters given.

    # Arguments
        x: 2D numpy array, single image.
        theta: Rotation angle in degrees.
        tx: Width shift.
        ty: Heigh shift.
        shear: Shear angle in degrees.
        zx: Zoom in x direction.
        zy: Zoom in y direction
        row_axis: Index of axis for rows in the input image.
        col_axis: Index of axis for columns in the input image.
        channel_axis: Index of axis for channels in the input image.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        The transformed version of the input.
    """
    if scipy is None:
        raise ImportError('Image transformations require SciPy. '
                          'Install SciPy.')
    transform_matrix = None
    if theta != 0:
        theta = np.deg2rad(theta)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        transform_matrix = rotation_matrix

    if tx != 0 or ty != 0:
        shift_matrix = np.array([[1, 0, tx],
                                 [0, 1, ty],
                                 [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shift_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shift_matrix)

    if shear != 0:
        shear = np.deg2rad(shear)
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shear_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shear_matrix)

    if zx != 1 or zy != 1:
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = zoom_matrix
        else:
            transform_matrix = np.dot(transform_matrix, zoom_matrix)

    if transform_matrix is not None:
        h, w = x.shape[row_axis], x.shape[col_axis]
        transform_matrix = transform_matrix_offset_center(
            transform_matrix, h, w)
        x = np.rollaxis(x, channel_axis, 0)
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]

        channel_images = [scipy.ndimage.interpolation.affine_transform(
            x_channel,
            final_affine_matrix,
            final_offset,
            order=1,
            mode=fill_mode,
            cval=cval) for x_channel in x]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_axis + 1)
    return x

def get_random_transform(img_shape,
                         rotation_range = 0,
                         height_shift_range = 0,
                         width_shift_range = 0,
                         shear_range = 0,
                         zoom_range = 0.,
                         horizontal_flip = False,
                         vertical_flip = False,
                         brightness_range = None,
                         fill_mode = 'constant',
                         cval = 0):
    """Generates random parameters for a transformation.

    # Arguments
        img_shape: Tuple of integers.
            Shape of the image that is transformed.

    # Returns
        A dictionary containing randomly chosen parameters describing the
        transformation.
    """
    img_row_axis = 0
    img_col_axis = 1

    if rotation_range:
        theta = np.random.uniform(
            -rotation_range,
            rotation_range)
    else:
        theta = 0

    if height_shift_range:
        try:  # 1-D array-like or int
            tx = np.random.choice(height_shift_range)
            tx *= np.random.choice([-1, 1])
        except ValueError:  # floating point
            tx = np.random.uniform(-height_shift_range,
                                   height_shift_range)
        if np.max(height_shift_range) < 1:
            tx *= img_shape[img_row_axis]
    else:
        tx = 0

    if width_shift_range:
        try:  # 1-D array-like or int
            ty = np.random.choice(width_shift_range)
            ty *= np.random.choice([-1, 1])
        except ValueError:  # floating point
            ty = np.random.uniform(-width_shift_range,
                                   width_shift_range)
        if np.max(width_shift_range) < 1:
            ty *= img_shape[img_col_axis]
    else:
        ty = 0

    if shear_range:
        shear = np.random.uniform(
            -shear_range,
            shear_range)
    else:
        shear = 0

    zoom_range = [1 - zoom_range, 1 + zoom_range]
    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(
            zoom_range[0],
            zoom_range[1],
            2)

    flip_horizontal = (np.random.random() < 0.5) * horizontal_flip
    flip_vertical = (np.random.random() < 0.5) * vertical_flip

    brightness = None
    if brightness_range is not None:
        if len(brightness_range) != 2:
            raise ValueError(
                '`brightness_range should be tuple or list of two floats. '
                'Received: %s' % (brightness_range,))
        brightness = np.random.uniform(brightness_range[0],
                                       brightness_range[1])

    transform_parameters = {'theta': theta,
                            'tx': tx,
                            'ty': ty,
                            'shear': shear,
                            'zx': zx,
                            'zy': zy,
                            'flip_horizontal': flip_horizontal,
                            'flip_vertical': flip_vertical,
                            'brightness': brightness,
                            'fill_mode': fill_mode,
                            'cval': cval}

    return transform_parameters

def apply_transform(x, transform_parameters):
    """Applies a transformation to an image according to given parameters.

    # Arguments
        x: 3D tensor, single image.
        transform_parameters: Dictionary with string - parameter pairs
            describing the transformation.
            Currently, the following parameters
            from the dictionary are used:
            - `'theta'`: Float. Rotation angle in degrees.
            - `'tx'`: Float. Shift in the x direction.
            - `'ty'`: Float. Shift in the y direction.
            - `'shear'`: Float. Shear angle in degrees.
            - `'zx'`: Float. Zoom in the x direction.
            - `'zy'`: Float. Zoom in the y direction.
            - `'flip_horizontal'`: Boolean. Horizontal flip.
            - `'flip_vertical'`: Boolean. Vertical flip.
            - `'channel_shift_intencity'`: Float. Channel shift intensity.
            - `'brightness'`: Float. Brightness shift intensity.

    # Returns
        A transformed version of the input (same shape).
    """
    # x is a single image, so it doesn't have image number at index 0
    img_row_axis = 0
    img_col_axis = 1
    img_channel_axis = 2

    x = apply_affine_transform(x, transform_parameters.get('theta', 0),
                               transform_parameters.get('tx', 0),
                               transform_parameters.get('ty', 0),
                               transform_parameters.get('shear', 0),
                               transform_parameters.get('zx', 1),
                               transform_parameters.get('zy', 1),
                               row_axis=img_row_axis,
                               col_axis=img_col_axis,
                               channel_axis=img_channel_axis,
                               fill_mode=transform_parameters['fill_mode'],
                               cval=transform_parameters['cval'])

    if transform_parameters.get('channel_shift_intensity') is not None:
        x = apply_channel_shift(x,
                                transform_parameters['channel_shift_intensity'],
                                img_channel_axis)

    if transform_parameters.get('flip_horizontal', False):
        x = flip_axis(x, img_col_axis)

    if transform_parameters.get('flip_vertical', False):
        x = flip_axis(x, img_row_axis)

    if transform_parameters.get('brightness') is not None:
        x = apply_brightness_shift(x, transform_parameters['brightness'])

    return x

def augment_image(x, aug_dict):
    return apply_transform(x, get_random_transform(x.shape, **aug_dict))