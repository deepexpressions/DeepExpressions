from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_file
from tensorflow.keras.backend import image_data_format

from .model_utils import _obtain_input_shape


# Download paths
VGGFACE_CE_L1_WEIGHTS_PATH = dict(
    name = "deep_expressions_tf_vggface_ce_l1_150.h5",
    path = ("https://github.com/deepexpressions/models/blob/master/"
            "emotions/deep_expressions_tf_vggface_ce_l1_150.h5?raw=true"),
    cache_subdir = "models/deep_expressions",
)
VGGFACE_CE_L3_WEIGHTS_PATH = dict(
    name = "deep_expressions_tf_vggface_ce_l3_150.h5",
    path = ("https://github.com/deepexpressions/models/blob/master/"
            "emotions/deep_expressions_tf_vggface_ce_l3_150.h5?raw=true"),
    cache_subdir = "models/deep_expressions",
)

VGGFACE_CK_L1_WEIGHTS_PATH = dict(
    name = "deep_expressions_tf_vggface_ck_l1_150.h5",
    path = ("https://github.com/deepexpressions/models/blob/master/"
            "emotions/deep_expressions_tf_vggface_ck_l1_150.h5?raw=true"),
    cache_subdir = "models/deep_expressions",
)
VGGFACE_CK_L3_WEIGHTS_PATH = dict(
    name = "deep_expressions_tf_vggface_ck_l3_150.h5",
    path = ("https://github.com/deepexpressions/models/blob/master/"
            "emotions/deep_expressions_tf_vggface_ck_l3_150.h5?raw=true"),
    cache_subdir = "models/deep_expressions",
)

VGGFACE_KDEF_L1_WEIGHTS_PATH = dict(
    name = "deep_expressions_tf_vggface_kdef_l1_150.h5",
    path = ("https://github.com/deepexpressions/models/blob/master/"
            "emotions/deep_expressions_tf_vggface_kdef_l1_150.h5?raw=true"),
    cache_subdir = "models/deep_expressions",
)
VGGFACE_KDEF_L3_WEIGHTS_PATH = dict(
    name = "deep_expressions_tf_vggface_kdef_l3_150.h5",
    path = ("https://github.com/deepexpressions/models/blob/master/"
            "emotions/deep_expressions_tf_vggface_kdef_l3_150.h5?raw=true"),
    cache_subdir = "models/deep_expressions",
)

VGGFACE_IFFG_L3_WEIGHTS_PATH = dict(
    name = "deep_expressions_tf_vggface_iffg_l3_150.h5",
    path = ("https://github.com/deepexpressions/models/blob/master/"
            "gestures/deep_expressions_tf_vggface_iffg_l3_150.h5?raw=true"),
    cache_subdir = "models/deep_expressions",
)


def _get_weights_path(weights, num_layers):
    """Return weights_path_dict based on weights name."""

    if weights == "ce" and num_layers == 1:
        return VGGFACE_CE_L1_WEIGHTS_PATH
    elif weights == "ce" and num_layers == 3:
        return VGGFACE_CE_L3_WEIGHTS_PATH

    elif weights in {"ck", "ck+"} and num_layers == 1:
        return VGGFACE_CK_L1_WEIGHTS_PATH
    elif weights in {"ck", "ck+"} and num_layers == 3:
        return VGGFACE_CK_L3_WEIGHTS_PATH

    if weights == "kdef" and num_layers == 1:
        return VGGFACE_KDEF_L1_WEIGHTS_PATH
    elif weights == "kdef" and num_layers == 3:
        return VGGFACE_KDEF_L3_WEIGHTS_PATH

    elif weights == "iffg" and num_layers == 3:
        return VGGFACE_IFFG_L3_WEIGHTS_PATH

    else:
        raise Exception("Couldn't find the model to download.")


def VGGFACE(weights=None, num_layers=1, num_classes=7, input_shape=(150, 150, 3)):
    """
    Load a FER Model trained on top of a VGGFACE implementation.

    Arguments:
        + weights -- Name of the database used to train the model. For details, 
        look at https://github.com/deepexpressions/models.
        + num_layers -- Number of layers in the neural network. Only 1 and 3 are available.
        + num_classes -- Number of output classes.
        + input_shape -- Input shape of the entrance image.

    Returns:
        + A model based on VGGFACE.
    """


    if not (weights in {"ce", "ck", "ck+", "kdef", "iffg", None} or os.path.exists(weights)):
        raise ValueError("The `weights` argument should be either "
                         "`None` (random initialization), `ce`, `ck`, `ck+`, `iffg`"
                         "or the path to the weights file to be loaded.")

    if weights in {"ce", "ck", "ck+"} and num_classes != 7:
        raise ValueError("If using `weights` as [`ce`, `ck`, `ck+`], "
                         "`num_classes` should be 7.")

    if weights == "iffg" and num_classes != 9:
        raise ValueError("If using `weights` as `iffg`, `num_classes` should be 9.")

    if weights in {"ce", "ck", "ck+", "kdef"} and input_shape != (150, 150, 3):
        warnings.warn("This model was trained with `input_shape = (150, 150, 3)`")

    if weights == "iffg" and input_shape != (224, 224, 3):
        warnings.warn("This model was trained with `input_shape = (224, 224, 3)`")


    # Determine model name
    model_name = "vggface" if weights is None else "vggface_" + weights


    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=150,
                                      min_size=48,
                                      data_format=image_data_format(),
                                      require_flatten=True,
                                      weights=weights)

    # Model input
    inputs = Input(shape=input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv1")(inputs)
    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv2")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv1")(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv2")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv1")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv2")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv3")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv1")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv2")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv3")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")(x)

    if num_layers == 1:
        # Classification Block
        x = MaxPooling2D((2, 2), strides=(2, 2), name="extra_pool")(x)
        x = Flatten(name="flatten")(x)
        x = Dropout(0.20, name="flatten/drop1")(x)
        x = Dense(7, activation="softmax", name="predictions")(x)
    
    elif num_layers == 3:
        # Classification Block
        x = Flatten(name="flatten")(x)
        x = Dense(256, activation="relu", name="fc1")(x)
        x = Dropout(0.3, name="fc1/drop1")(x)
        x = Dense(128, activation="relu", name="fc2")(x)
        x = Dropout(0.2, name="fc2/drop1")(x)
        x = Dense(64, activation="relu", name="fc3")(x)
        x = Dense(num_classes, activation="softmax", name="predictions")(x)

    else:
        raise ValueError("`num_layers` must be 1 or 3.")

    # Create model
    model = Model(inputs, x, name=model_name)

    # Load weights
    if weights in {"ce", "ck", "ck+", "kdef", "iffg"}:
        wpd = _get_weights_path(weights, num_layers)
        weights_path = get_file(wpd["name"], wpd["path"], cache_subdir=wpd["cache_subdir"])
        model.load_weights(weights_path)

    elif weights is not None:
        model.load_weights(weights)

    return model