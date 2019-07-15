from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
import PIL.Image as Image
from tensorflow.keras.applications import densenet, mobilenet, resnet50


_preprocess_by_convnet_dict = dict(
    vgg16 = (resnet50.preprocess_input, "tensor"),
    vgg19 = (resnet50.preprocess_input, "tensor"),
    resnet50 = (resnet50.preprocess_input, "tensor"),
    # densenet = (densenet.preprocess_input, "numpy"),
    nasnet = (mobilenet.preprocess_input, "numpy"),
    xception = (mobilenet.preprocess_input, "numpy"),
    mobilenet = (mobilenet.preprocess_input, "numpy"),
    mobilenet_v2 = (mobilenet.preprocess_input, "numpy"),
    inception_v3 = (mobilenet.preprocess_input, "numpy"),
    inception_resnet_v2 = (mobilenet.preprocess_input, "numpy"),
)


def imread(image_path, as_array=False, normalize=None):
    """
    Read image file.

    Arguments:
        + image_path (str) -- Path to image file.
        + as_array (bool) -- Return numpy array instead of tf.Tensor.
        + normalize (str) -- If "norm_b": normalize image to range [0, 1].
                             If "norm_n": normalize image to range [-1, 1].
                             Else do not normalize.

    Output:
        + An image (as Tensor or array)
    """

    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)

    image = image/255 if normalize == "norm_p" else ((
        tf.dtypes.cast(image, tf.float32)/127.5)-1) if normalize == "norm_n" else image

    image = image if not as_array else image.numpy()
    return image


def preprocess_by_convnet(image, convnet="vgg16", as_array=False):
    if convnet not in _preprocess_by_convnet_dict.keys():
        raise Exception("Unkown ConvNet.")

    # Get preprocess and input type
    preprocess_input, input_type = _preprocess_by_convnet_dict[convnet]
    # Convert image to float32
    image = tf.dtypes.cast(image, tf.float32)
    # Preprocess image
    image = preprocess_input(image)

    if input_type == "tensor":
        image = tf.dtypes.cast(image, tf.uint8)

    if as_array:
        return image.numpy()

    else:
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