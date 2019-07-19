from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import tensorflow as tf

from . import model_utils


class FERModel():
    """
    Facial Expression Recognition Model.

    Arguments:
        + model_name (str) -- Name of the model. Options available at : http://github.com/deepexpressions/models.
        + filename (str) -- Path to custom model saved (only *.h5 files).

    Methods:
        + predict -- Predict facial expressions from input_data.
        + summary -- Prints a summary representation of the model loaded.
        
    Attributes:
        + model_name -- name of the model.
        + model -- Keras Sequential model.
    """


    def __init__(self, name, labels=None, input_shape=(150, 150, 3)):
        """Model constructor."""

        self.name = name
        self.input_shape = input_shape

        # Load DeepExpressions trained model
        if not os.path.exists(name):
            self.model = model_utils.load_model(name, input_shape)
            self.labels = labels if labels is not None else model_utils._get_model_labels(name)
            self._tm = True # Use to know if the loaded model is from
                            # DeepExpressions or local file

        # Load local model
        else:
            if labels is None:
                raise Exception("Can't use local file as model without define its `labels`")

            self.model = model_utils.load_model_from_dir(name)
            self.labels = labels
            self._tm = False


    def predict(self, input_data, return_labels=False, return_scores=False):
        """
        Predict facial expressions from input data.

        Arguments:
            + input_data -- Images to predict. It should be preprocessed.
            + return_labels (bool) -- Returns predictions with label names.
            + return_scores (bool) -- Returns scores from each class in the prediction.

        Output:
            + (list of predictions, [labeled predictions], [scores])
        """
        
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


    def preprocess_input(self, input_data, normalize=None):
        """
        Preprocess input_data before predictions.

        Arguments:
            + input_data -- Images to predict. It can be np.ndarray, tf.Tensor or a list of the previous types.
            + normalize (str) -- If "norm_b": normalize image to range [0, 1].
                                 If "norm_n": normalize image to range [-1, 1].
                                 Else do not normalize.

        Output:
            + batch of preprocessed images.
        """

        h, w = self.input_shape[:2]

        # preprocess_func = {norm_p, norm_n, None}
        preprocess_func = lambda x: x/255 if normalize == "norm_p" else (
            (tf.dtypes.cast(x, tf.float32)/127.5)-1) if normalize == "norm_n" else x

        # input_data is a single image
        if isinstance(input_data, (np.ndarray, tf.Tensor)):
            input_data = tf.image.resize([input_data], size=(h, w))
            input_data = preprocess_func(input_data)

        # input_data are multiple images
        elif isinstance(input_data, list):
            images = []

            for image in input_data:
                images.append(
                    tf.image.resize(image, size=[h, w])
                )

            input_data = tf.convert_to_tensor(images, dtype=tf.float32)
            input_data = preprocess_func(input_data)

        else:
            raise "Invalid input_data"

        return input_data


    def info(self):
        """Prints DeepExpressions model informations."""

        if not self._tm:
            print("Local file loaded. No information could be found.")

        else:
            model_utils.print_model_info(self.name)


    def summary(self):
        """Prints a summary representation of the model loaded."""
        self.model.summary()


    def _label_predictions(self, x): 
        """Convert one-hot-encoded predictions to label names."""
        return self.labels[x]


