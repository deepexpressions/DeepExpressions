from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
import PIL.Image as Image


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


def imsave(image, filename):
    """
    Save image.

    Arguments:
        + image (np.ndarray, tf.Tensor) -- Image to be saved.
        + filename (str) -- Name of the file to be saved.
    """

    # Convert from tf.Tensor to np.ndarray
    if isinstance(image, tf.Tensor):
        image = image.numpy()
    # Convert from np.ndarray to PIL.Image object
    if isinstance(image, np.ndarray):
        image = Image.fromarray(np.uint8(image)).convert("RGB")
    # Save image
    image.save(filename)