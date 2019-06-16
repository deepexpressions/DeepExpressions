from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf


def imread(image_path, normalize=False):
    """
    Read image file.

    Arguments:
        + image_path (str) -- Path to image file.
        + normalize (bool) -- Normalize image to range [0, 1].

    Output:
        + An image as Tensor
    """

    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = image if not normalize else image/255
    return image


def imresize(image, height=128, width=128):
    """
    Resize image.

    Arguments:
        + image (tf.Tensor) -- Image to be resized
        + height (int) -- New image height.
        + width (int) -- New image width.

        Output:
        + A resized image
    """
    return tf.image.resize(image, [height, width])