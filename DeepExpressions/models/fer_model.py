from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import tensorflow as tf

from . import utils
from .applications import *


# Image size from input_data
IMAGE_SIZE = 128


class FERModel():
    """
    Facial Expression Recognition Model.

    Arguments:
        + model_name (str) -- Name of the model. Options available at : http://github.com/deepexpressions/models.
        + filename (str) -- Path to custom model saved (only *.h5 files).

    Methods:
        + predict -- Predict facial expressions from input_data.
        + summary -- Prints a summary representation of the model loaded.
        + train_info -- Prints some training information about the model loaded.
        
    Attributes:
        + model_name -- name of the model.
        + model -- Keras Sequential model.
    """


    def __init__(self, name=None, labels=None, input_shape=(128, 128, 3), filename=None):
        """Model constructor."""

        if (name == filename == None) or (name is not None and filename is not None):
            raise Exception("None or more than one model specified. Please select a model name or a path.")

        if name is not None:
            if not utils._model_exists(name):
                raise Exception("Model name does not exists. Try another.")
            else:
                self.model = utils.load_model(name)

        else:
            self.model = utils.load_model_from_dir(filename)


    def predict(self, input_data, return_labels=False, return_scores=False):
        """
        Predict facial expressions from input data.

        Arguments:
            + input_data -- Images to predict. It can be np.ndarray, tf.Tensor or a list of the previous types.
            + return_labels (bool) -- Returns predictions with label names.
            + return_scores (bool) -- Returns scores from each class in the prediction.

        Output:
            + (list of predictions, [labeled predictions], [scores])
        """
        # pre-process input_data
        input_data = self._preprocess_input_data(input_data)

        # Predict data
        scores = self.model.predict(input_data)
        predictions = tf.argmax(scores, axis=1)

        output = (predictions, )

        if return_labels:
            labeled_predictions = list(map(self._label_predictions, predictions.numpy().tolist()))
            output += (labeled_predictions, )

        if return_scores:
            output += (scores, )

        return output


    def train_info(self):
        """Prints some training information about the model loaded."""
        if self._filename is not None:
            raise "Can't find informations about custom models."
        print(FER_MODELS[self.model_name]["info"])


    def summary(self):
        """Prints a summary representation of the model loaded."""
        self.model.summary()


    def _preprocess_input_data(self, input_data):
        """Pre process input data before it."""

        # input_data is a single image and is an array
        if isinstance(input_data, np.ndarray):
            input_data = tf.image.resize(
                [input_data], size=(IMAGE_SIZE, IMAGE_SIZE))
            input_data /= 255

        # input_data is a single image and is a Tensor
        elif isinstance(input_data, tf.Tensor):
            input_data = tf.image.resize(
                [input_data], size=(IMAGE_SIZE, IMAGE_SIZE))
            input_data /= 255

        # input_data are multiple images
        elif isinstance(input_data, list):
            images = []

            for image in input_data:
                images.append(
                    tf.image.resize(image, size=[IMAGE_SIZE, IMAGE_SIZE])
                )

            input_data = tf.convert_to_tensor(images, dtype=tf.float32)
            input_data /= 255

        else:
            raise "Invalid input_data"

        return input_data


    def _label_predictions(self, x): 
        """Convert one-hot-encoded predictions to label names."""
        return FER_LABELS[x]


