from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pathlib
import tensorflow as tf
# from matplotlib import pyplot as plt


AUTOTUNE = tf.data.experimental.AUTOTUNE


class DataLoader():
    """
    DataLoader is an instance to load image data from a folder using tensorflow>=2.0.0-alpha0.

    Arguments:
        + data_root (str) -- Path to dataset folder.
        + image_size (int) -- Size to resize the loaded images.
        + data_root (str) -- Directory containing sub-folder for each label.
        + batch_size (int) -- Dataset batch size.
        + data_augm (list) -- List of data augmentations techniques.
        + steps_per_epoch (int) -- Number of steps per epoch.
        + cache (bool) -- Load dataset in cache.

    Methods:
        + show_sample_batch -- show data from the first batch.

    Attributes:
        + ds -- dataset in batches.
        + image_count -- number of images from dataset.
        + label_names -- list with labels.
        + label_to_index -- dict with indexed labels.
    """


    def __init__(self, data_root, image_size=128, batch_size=32, data_augm=None, steps_per_epoch=None, cache=False):
        """DataLoader constructor."""

        # Pass input variables
        self.batch_size = batch_size
        self.image_size = image_size

        # Load all images and their labels
        all_image_paths, all_image_labels = self._load_all_images_and_labels(
            data_root)

        # Get the number of images loaded
        self.image_count = len(all_image_paths)

        # Pass steps_per_epoch variable
        if steps_per_epoch is None:
            self.steps_per_epoch = self._calc_steps_per_epoch()
        else:
            self.steps_per_epoch = steps_per_epoch

        # Load and preprocess the data
        ds = tf.data.Dataset.from_tensor_slices(
            (all_image_paths, all_image_labels))
        ds = ds.map(
            self._load_and_preprocess_from_path_label, num_parallel_calls=AUTOTUNE)

        # Apply data augmentation
        if data_augm is not None:
            ds_augm = ds

            for augmentation in data_augm:
                ds_augm_aux = ds.map(augmentation, num_parallel_calls=AUTOTUNE)
                ds_augm = ds_augm.concatenate(ds_augm_aux)

            ds = ds_augm
            self.image_count *= (1 + len(data_augm))

        # Prepare the dataset
        if cache:
            ds = ds.cache()

        # ds = ds.apply(
        #     tf.data.experimental.shuffle_and_repeat(buffer_size=self.image_count))
        ds = ds.shuffle(buffer_size=self.image_count).repeat()
        self.ds = ds.batch(self.batch_size).prefetch(buffer_size=AUTOTUNE)


    # def show_sample_batch(self, subplot_shape=None, figsize=(5, 5), display_title=None):
    #     """
    #     Show random images and their labels from dataset.

    #     Arguments:
    #         + subplots_shape (tuple) -- Shape of figures subplot. If None, shape = (1, batch_size).
    #         + figsize (tuple) -- Size of the figure.
    #         + display_title (str) -- Figure title.
    #     """

    #     if subplot_shape is None:
    #         ss0, ss1 = 1, self.batch_size
    #         num_images = self.batch_size
    #     else:
    #         ss0, ss1 = subplot_shape
    #         num_images = ss0 * ss1

    #     plt.figure(figsize=figsize)

    #     for images, labels in self.ds.take(1):
    #         for n, (image, label) in enumerate(zip(images, labels)):
    #             if n+1 > num_images:
    #                 break

    #             plt.subplot(ss0, ss1, n+1)
    #             plt.imshow(image)
    #             plt.title(self.label_names[label.numpy()])
    #             plt.xticks([])
    #             plt.yticks([])
    #     plt.suptitle(display_title)
    #     plt.show()


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


    def _preprocess_image(self, image):
        """Pre-process loaded images."""
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [self.image_size, self.image_size])
        image /= 255.0
        return image


    def _load_and_preprocess_image(self, image_path):
        """Load and pre-process images."""
        image = tf.io.read_file(image_path)
        return self._preprocess_image(image)


    def _load_and_preprocess_from_path_label(self, path, label):
        """Load and pre-process images and its respectives labels."""
        return self._load_and_preprocess_image(path), label


    def _calc_steps_per_epoch(self):
        """Calculate steps_per_epoch if none is given."""
        return tf.math.ceil(self.image_count/self.batch_size).numpy()