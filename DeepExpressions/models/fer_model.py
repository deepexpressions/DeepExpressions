from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from .fer_models_info import FER_MODELS, FER_LABELS


# Image size from input_data
IMAGE_SIZE = 128


class FERModel():
    """
    Facial Expression Recognition Model.

    Arguments:
        + model_name (str) -- Name of the model. Options available at : http://github.com/deepexpressions/models.
        + custom_model (str) -- Path to custom model saved (only *.h5 files).

    Methods:
        + predict -- Predict facial expressions from input_data.
        + summary -- Prints a summary representation of the model loaded.
        + train_info -- Prints some training information about the model loaded.
        
    Attributes:
        + model_name -- name of the model.
        + model -- Keras Sequential model.
    """


    def __init__(self, model_name="ce-xception-512-256", custom_model=None):
        """Model constructor."""

        self.model_name = model_name
        self._custom_model = custom_model
        self.model = load_model(model_name, custom_model)


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
        if self._custom_model is not None:
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


def load_model(model_name="ce-xception-512-256", custom_model=None):
    """
    Load a Facial Expression Recognition classifier from DeepExpressions models.

    Arguments:
        + model_name (str) -- Name of the model. Options available at : ?.
        + custom_model (str) -- Path to custom model saved (only *.h5 files).

    Output:
        + Keras Sequential model for acial Expression Recognition.
    """
    def maybe_download_model():
        # Get info from the model
        model = FER_MODELS[model_name]
        # Use keras utils to download model
        model_path = tf.keras.utils.get_file(
            model["filename"],
            model["url"],
            cache_subdir="deep_expressions"
        )
        return model_path

    print("[INFO] Loading model '{}' ...".format(model_name), end=" ")
    
    if custom_model is None:
        # Path to DeepExpression model
        model_path = maybe_download_model()
    else:
        # Verify file extension
        assert custom_model.endswith(".h5"), "Custom model must be save with extenstion '.h5'."
        # Path to custom model
        model_path = custom_model
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    print("Done.")

    return model