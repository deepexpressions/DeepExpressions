from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pathlib
import tensorflow as tf
from functools import partial
from . import data_augmentation as daug


# Autotune
AUTOTUNE = tf.data.experimental.AUTOTUNE

# Norm P [0., 1.]
_norm_p = lambda image, label: (image / 255.0, label)
# Norm P [-1., 1.]
_norm_n = lambda image, label: ((image / 127.5) - 1, label)
# Clip P [0., 1.]
_clip_p = lambda image, label: (tf.clip_by_value(image, 0.0, 1.0), label)
# Clip P [-1., 1.]
_clip_n = lambda image, label: (tf.clip_by_value(image, -1.0, 1.0), label)


class DataLoader():
    """
    DataLoader is an instance to load image data from a folder using tensorflow>=2.0.0-alpha0.

    Arguments:
        + data_root (str) -- Path to dataset folder.
        + image_size (int) -- Size to resize the loaded images.
        + data_root (str) -- Directory containing sub-folder for each label.
        + batch_size (int) -- Dataset batch size.
        + steps_per_epoch (int) -- Number of steps per epoch.
        + cache (bool) -- Load dataset in cache.
        + normalize (str) -- If "norm_b": normalize image to range [0, 1].
                             If "norm_n": normalize image to range [-1, 1].
                             Else do not normalize.
        + flip_left_right (bool) -- Data augmentation: Flip an image horizontally (left to right).
        + flip_up_down (bool) -- Data augmentation: Flip an image vertically (upside down).
        + random_brightness (float) -- Data augmentation: Adjust the brightness of images by a random factor.
        + random_contrast (tuple) -- Data augmentation: Adjust the contrast of an image or images by a random factor.
        + random_hue (float) -- Data augmentation: Adjust the hue of RGB images by a random factor.
        + random_jpeg_quality (tuple) -- Data augmentation: Randomly changes jpeg encoding quality for inducing jpeg noise.
        + random_rotation (float) -- Data augmentation: Performs a random rotation of a image.
        + random_saturation (tuple) -- Data augmentation: Adjust the saturation of RGB images by a random factor.
        + random_shear (float) -- Data augmentation: Performs a random spatial shear of a image.
        + random_zoom (tuple) -- Data augmentation: Performs a random spatial zoom of a image.

    Methods:
        + show_sample_batch -- show data from the first batch.

    Attributes:
        + ds -- dataset in batches.
        + image_count -- number of images from dataset.
        + label_names -- list with labels.
        + label_to_index -- dict with indexed labels.
    """


    def __init__(self, data_root, image_size=128, batch_size=32, 
        steps_per_epoch=None, cache=False, normalize=None, 
        flip_left_right=False, flip_up_down=False,
        random_brightness=None, random_contrast=None,
        random_hue=None, random_jpeg_quality=None,
        random_rotation=None, random_saturation=None,
        random_shear=None, random_zoom=None):

        # Pass input variables
        self.batch_size = batch_size
        self.image_size = image_size

        # Load all images and their labels
        all_image_paths, all_image_labels = self._load_all_images_and_labels(
            data_root)

        # Get the number of images loaded
        self.image_count = len(all_image_paths)
        _original_image_count = len(all_image_paths)

        # Pass steps_per_epoch variable
        if steps_per_epoch is None:
            self.steps_per_epoch = self._calc_steps_per_epoch()
        else:
            self.steps_per_epoch = steps_per_epoch

        # Load the data
        ds = tf.data.Dataset.from_tensor_slices(
            (all_image_paths, all_image_labels))
        ds = ds.map(
            self._load_and_preprocess_from_path_label, num_parallel_calls=AUTOTUNE)

        # Copy dataset for augment original only
        ds_augm = ds

        # Apply data augmentation
        if flip_left_right:
            ds_augm_aux = ds.map(daug.flip_left_right, num_parallel_calls=AUTOTUNE)
            ds_augm = ds_augm.concatenate(ds_augm_aux)
            self.image_count += _original_image_count

        if flip_up_down:
            ds_augm_aux = ds.map(daug.flip_up_down, num_parallel_calls=AUTOTUNE)
            ds_augm = ds_augm.concatenate(ds_augm_aux)
            self.image_count += _original_image_count

        if random_brightness is not None:
            if not (isinstance(random_brightness, int) or isinstance(random_brightness, float)):
                raise Exception("random_brightness must be and integer or float.")
            
            daug_func = partial(daug.random_brightness, max_delta=random_brightness)
            ds_augm_aux = ds.map(daug_func, num_parallel_calls=AUTOTUNE)
            ds_augm = ds_augm.concatenate(ds_augm_aux)
            self.image_count += _original_image_count

        if random_contrast is not None:
            if not (len(random_contrast) !=2 or len([False for v in random_contrast if not(
                isinstance(v, int) or isinstance(v, float))])==0):
                raise Exception("random_contrast must be a 2 values tuple of integer or floats")

            lower, upper = random_contrast
            daug_func = partial(daug.random_contrast, lower=lower, upper=upper)
            ds_augm_aux = ds.map(daug_func, num_parallel_calls=AUTOTUNE)
            ds_augm = ds_augm.concatenate(ds_augm_aux)
            self.image_count += _original_image_count

        if random_hue is not None:
            if not (isinstance(random_hue, int) or isinstance(random_hue, float)):
                raise Exception("random_hue must be and integer or float.")
            
            daug_func = partial(daug.random_hue, max_delta=random_hue)
            ds_augm_aux = ds.map(daug_func, num_parallel_calls=AUTOTUNE)
            ds_augm = ds_augm.concatenate(ds_augm_aux)
            self.image_count += _original_image_count

        if random_jpeg_quality is not None:
            if not (len(random_jpeg_quality) !=2 or len([False for v in random_jpeg_quality if not(
                isinstance(v, int) or isinstance(v, float))])==0):
                raise Exception("random_jpeg_quality must be a 2 values tuple of integer or floats")

            lower, upper = random_jpeg_quality
            daug_func = partial(daug.random_jpeg_quality, min_jpeg_quality=lower, max_jpeg_quality=upper)
            ds_augm_aux = ds.map(daug_func, num_parallel_calls=AUTOTUNE)
            ds_augm = ds_augm.concatenate(ds_augm_aux)
            self.image_count += _original_image_count

        if random_rotation is not None:
            if not (isinstance(random_rotation, int) or isinstance(random_rotation, float)):
                raise Exception("random_rotation must be and integer or float.")
            
            daug_func = partial(daug.random_rotation, angle=random_rotation)
            ds_augm_aux = ds.map(daug_func, num_parallel_calls=AUTOTUNE)
            ds_augm = ds_augm.concatenate(ds_augm_aux)
            self.image_count += _original_image_count

        if random_saturation is not None:
            if not (len(random_saturation) !=2 or len([False for v in random_saturation if not(
                isinstance(v, int) or isinstance(v, float))])==0):
                raise Exception("random_saturation must be a 2 values tuple of integer or floats")

            lower, upper = random_saturation
            daug_func = partial(daug.random_saturation, lower=lower, upper=upper)
            ds_augm_aux = ds.map(daug_func, num_parallel_calls=AUTOTUNE)
            ds_augm = ds_augm.concatenate(ds_augm_aux)
            self.image_count += _original_image_count

        if random_shear is not None:
            if not (isinstance(random_shear, int) or isinstance(random_shear, float)):
                raise Exception("random_shear must be and integer or float.")
            
            daug_func = partial(daug.random_shear, intensity=random_shear)
            ds_augm_aux = ds.map(daug_func, num_parallel_calls=AUTOTUNE)
            ds_augm = ds_augm.concatenate(ds_augm_aux)
            self.image_count += _original_image_count

        if random_zoom is not None:
            if not (len(random_zoom) !=2 or len([False for v in random_zoom if not(
                isinstance(v, int) or isinstance(v, float))])==0):
                raise Exception("random_zoom must be a 2 values tuple of integer or floats")

            daug_func = partial(daug.random_zoom, zoom_range=random_zoom)
            ds_augm_aux = ds.map(daug_func, num_parallel_calls=AUTOTUNE)
            ds_augm = ds_augm.concatenate(ds_augm_aux)
            self.image_count += _original_image_count

        # Replace origal dataset for the augmented one
        ds = ds_augm

        ## Preprocess dataset
        # By norm P [0,1]
        if normalize == "norm_p":
            ds = ds.map(_norm_p, num_parallel_calls=AUTOTUNE)
            ds = ds.map(_clip_p, num_parallel_calls=AUTOTUNE)
        # By norm N [-1,1]
        elif normalize == "norm_n":
            ds = ds.map(_norm_n, num_parallel_calls=AUTOTUNE)
            ds = ds.map(_clip_n, num_parallel_calls=AUTOTUNE)

        # Cache dataset
        if cache:
            ds = ds.cache()

        # Shuffle and batch dataset
        ds = ds.shuffle(buffer_size=self.image_count).repeat()
        self.ds = ds.batch(self.batch_size).prefetch(buffer_size=AUTOTUNE)


    def _load_all_images_and_labels(self, data_root):
        data_root = pathlib.Path(data_root)

        # List the available data paths
        all_image_paths = [str(p) for p in list(data_root.glob("*/*"))]
        # List the available labels
        self.label_names = sorted(
            item.name for item in data_root.glob("*/") if item.is_dir())
        # Assign an index to each label:
        self.label_to_index = dict((name, index)
                                   for index, name in enumerate(self.label_names))
        # Create a list of every file, and its label index
        all_image_labels = [self.label_to_index[pathlib.Path(
            path).parent.name] for path in all_image_paths]

        return all_image_paths, all_image_labels


    def _decode_and_resize_image(self, image):
        """Pre-process loaded images."""
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [self.image_size, self.image_size])
        return image


    def _load_and_preprocess_image(self, image_path):
        """Load and pre-process images."""
        image = tf.io.read_file(image_path)
        return self._decode_and_resize_image(image)


    def _load_and_preprocess_from_path_label(self, path, label):
        """Load and pre-process images and its respectives labels."""
        return self._load_and_preprocess_image(path), label


    def _calc_steps_per_epoch(self):
        """Calculate steps_per_epoch if none is given."""
        return tf.math.ceil(self.image_count/self.batch_size).numpy()