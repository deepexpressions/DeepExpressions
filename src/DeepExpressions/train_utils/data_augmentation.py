import tensorflow as tf
from functools import partial


def flip_left_right(image, label: tuple):
    """
    Data augmentation: Flip an image horizontally (left to right).
    
    Arguments:
        + image (tf.Tensor) -- Image to transform.
        + label (tf.Tensor) -- Label from the image.
    """
    return tf.image.flip_left_right(image), label


def flip_up_down(image, label: tuple):
    """
    Data augmentation: Flip an image vertically (upside down).
    
    Arguments:
        + image (tf.Tensor) -- Image to transform.
        + label (tf.Tensor) -- Label from the image.
    """
    return tf.image.flip_up_down(image), label


def random_brightness(image, label: tuple, max_delta=0.5):
    """
    Data augmentation: Adjust the brightness of images by a random factor.
    
    Arguments:
        + image (tf.Tensor) -- Image to transform.
        + label (tf.Tensor) -- Label from the image.
        + max_delta (float) -- Amount to add to the pixel values.
    """
    tf_image = tf.image.random_brightness(image, max_delta=max_delta)
    return tf_image, label


def random_contrast(image, label: tuple, lower=0.5, upper=2.5):
    """
    Data augmentation: Adjust the contrast of an image or images by a random factor.
    
    Arguments:
        + image (tf.Tensor) -- Image to transform.
        + label (tf.Tensor) -- Label from the image.
        + lower (float) -- Lower bound for the random contrast factor.
        + upper (float) -- Upper bound for the random contrast factor.
    """
    tf_image = tf.image.random_contrast(image, lower=lower, upper=upper)
    return tf_image, label


def random_hue(image, label: tuple, max_delta=0.5):
    """
    Data augmentation: Adjust the hue of RGB images by a random factor.

    Arguments:
        + image (tf.Tensor) -- Image to transform.
        + label (tf.Tensor) -- Label from the image.
        + max_delta (float) -- Maximum value for the random delta.
    """
    return tf.image.random_hue(image, max_delta=max_delta), label


def random_jpeg_quality(image, label: tuple, min_jpeg_quality=5, max_jpeg_quality=25):
    """
    Data augmentation: Randomly changes jpeg encoding quality for inducing jpeg noise.
    
    Arguments:
        + image (tf.Tensor) -- Image to transform.
        + label (tf.Tensor) -- Label from the image.
        + min_jpeg_quality () -- Minimum jpeg encoding quality to use.
        + max_jpeg_quality () -- Maximum jpeg encoding quality to use.
    """
    return tf.image.random_jpeg_quality(image, min_jpeg_quality=min_jpeg_quality, 
        max_jpeg_quality=max_jpeg_quality), label


def random_rotation(image, label: tuple, angle=45):
    """
    Data augmentation: Performs a random rotation of a image.

    Arguments:
        + image (tf.Tensor) -- Image to transform.
        + label (tf.Tensor) -- Label from the image.
        + angle (int) -- Rotation range, in degrees.
    """
    return tf.py_function(partial(_random_rotation, angle=angle), [image], tf.float32), label


def _random_rotation(image, angle=45):
    """
    Keras transformation: Performs a random rotation of a image.

    Arguments:
        + image (tf.Tensor) -- Image to transform.
        + angle (int) -- Rotation range, in degrees.
    """
    return tf.keras.preprocessing.image.random_rotation(image.numpy(), angle, 0, 1, 2)


def random_saturation(image, label: tuple, lower=0, upper=5):
    """
    Data augmentation: Adjust the saturation of RGB images by a random factor.
    
    Arguments:
        + image (tf.Tensor) -- Image to transform.
        + label (tf.Tensor) -- Label from the image.
        + lower (float) -- Lower bound for the random saturation factor.
        + upper (float) -- Upper bound for the random saturation factor.
    """
    return tf.image.random_saturation(image, lower=lower, upper=upper), label


def random_shear(image, label: tuple, intensity=45):
    """
    Data augmentation: Performs a random spatial shear of a image.
    
    Arguments:
        + image (tf.Tensor) -- Image to transform.
        + label (tf.Tensor) -- Label from the image.
        + intensity (int) -- Transformation intensity in degrees.
    """
    return tf.py_function(partial(_random_shear, intensity=intensity), [image], tf.float32), label


def _random_shear(image, intensity=45):
    """
    Keras transformation: Performs a random spatial shear of a image.
    
    Arguments:
        + image (tf.Tensor) -- Image to transform.
        + intensity (int) -- Transformation intensity in degrees.
    """
    return tf.keras.preprocessing.image.random_shear(image.numpy(), intensity, 0, 1, 2)
    

def random_zoom(image, label: tuple, zoom_range=(0.6, 0.6)):
    """
    Data augmentation: Performs a random spatial zoom of a image.

    Arguments:
        + image (tf.Tensor) -- Image to transform.
        + label (tf.Tensor) -- Label from the image.
        + zoom_range (Tuple of floats) -- Zoom range for width and height.
    """
    return tf.py_function(partial(_random_zoom, zoom_range=zoom_range), [image], tf.float32), label


def _random_zoom(image, zoom_range=(0.6, 0.6)):
    """
    Keras transformation: Performs a random spatial zoom of a image.
    
    Arguments:
        + image (tf.Tensor) -- Image to transform.
        + zoom_range (Tuple of floats) -- Zoom range for width and height.
    """
    return tf.keras.preprocessing.image.random_zoom(image.numpy(), zoom_range, 0, 1, 2)