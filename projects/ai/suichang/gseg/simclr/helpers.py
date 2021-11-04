import tensorflow as tf
import numpy as np
# from augmentation.gaussian_filter import GaussianBlur


def get_negative_mask(batch_size):
    # return a mask that removes the similarity score of equal/similar images.
    # this function ensures that only distinct pair of images get their similarity scores
    # passed as negative examples
    negative_mask = np.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0
    return tf.constant(negative_mask)


# def gaussian_filter(v1, v2):
#     k_size = int(v1.shape[1] * 0.1)  # kernel size is set to be 10% of the image height/width
#     gaussian_ope = GaussianBlur(kernel_size=k_size, min=0.1, max=2.0)
#     [v1, ] = tf.py_function(gaussian_ope, [v1], [tf.float32])
#     [v2, ] = tf.py_function(gaussian_ope, [v2], [tf.float32])
#     return v1, v2