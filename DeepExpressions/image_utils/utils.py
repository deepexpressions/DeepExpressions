from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf


def imread(image_path, as_array=False, normalize=False):
    """
    Read image file.

    Arguments:
        + image_path (str) -- Path to image file.
        + as_array (bool) -- Return numpy array instead of tf.Tensor.
        + normalize (bool) -- Normalize image to range [0, 1].

    Output:
        + An image (as Tensor or array)
    """

    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = image if not normalize else image/255
    image = image if not as_array else image.numpy()
    return image


def imresize(image, height=128, width=128, as_array=False):
    """
    Resize image.

    Arguments:
        + image (tf.Tensor) -- Image to be resized
        + height (int) -- New image height.
        + width (int) -- New image width.
        + as_array (bool) -- Return numpy array instead of tf.Tensor.

        Output:
        + A resized image (as Tensor or array)
    """
    image = tf.image.resize(image, [height, width]) 
    image = image if not as_array else image.numpy()
    return image