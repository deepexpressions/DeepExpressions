from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_file
from .applications_utils import _obtain_input_shape


# Keras VGG16 Implementation
# https://github.com/keras-team/keras-applications/
VGG16_WEIGHTS_PATH = dict(
    name = "vgg16_weights_tf_dim_ordering_tf_kernels.h5",
    path = ("https://github.com/fchollet/deep-learning-models/releases/"
            "download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5"),
    cache_subdir = "models",
)
VGG16_WEIGHTS_PATH_NO_TOP = dict(
    name = "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",
    path = ("https://github.com/fchollet/deep-learning-models/releases/"
            "download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"),
    cache_subdir = "models",
)


# Oxford VGGFace Implementation using Keras Functional Framework v2+
# https://github.com/rcmalli/keras-vggface
VGG_FACE_WEIGHTS_PATH = dict(
    name = "rcmalli_vggface_tf_vgg16.h5",
    path = ("https://github.com/rcmalli/keras-vggface/releases/"
            "download/v2.0/rcmalli_vggface_tf_vgg16.h5"),
    cache_subdir = "models/vgg_face",
)
VGG_FACE_WEIGHTS_PATH_NO_TOP = dict(
    name = "rcmalli_vggface_tf_notop_vgg16.h5",
    path = ("https://github.com/rcmalli/keras-vggface/releases/"
            "download/v2.0/rcmalli_vggface_tf_notop_vgg16.h5"),
    cache_subdir = "models/vgg_face",
)


def _get_weights_path(weights, include_top):
    if weights == "vggface":
        weights_path = VGG_FACE_WEIGHTS_PATH if include_top else VGG_FACE_WEIGHTS_PATH_NO_TOP
    
    elif weights == "imagenet":
        weights_path = VGG16_WEIGHTS_PATH if include_top else VGG16_WEIGHTS_PATH_NO_TOP

    return weights_path


def VGG16(include_top=False, weights=None, input_shape=(224, 224, 3),
          pooling=None, num_classes=1000, model_name="vgg16"):


    if not (weights in {"imagenet", "vggface", None} or os.path.exists(weights)):
        raise ValueError("The `weights` argument should be either "
                         "`None` (random initialization), `imagenet`, `vggface` "
                         "or the path to the weights file to be loaded.")

    if weights == "imagenet" and include_top and classes != 1000:
        raise ValueError("If using `weights` as `imagenet` with `include_top` "
                         "as true, `classes` should be 1000")

    if weights == "vggface" and include_top and classes != 2622:
        raise ValueError("If using `weights` as `imagenet` with `include_top` "
                         "as true, `classes` should be 2622")


    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48 if weights == "vggface" else 32,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
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

    # Block 5
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv1")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv2")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv3")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool")(x)

    if include_top:
        # Classification Block
        x = Flatten(name="flatten")(x)
        x = Dense(4096, activation="relu", name="fc1")(x)
        x = Dense(4096, activation="relu", name="fc2")(x)
        x = Dense(num_classes, activation="softmax", name="predictions")(x)

    else:
        if pooling == "avg":
            x = GlobalAveragePooling2D()(x)
        elif pooling == "max":
            x = GlobalMaxPooling2D()(x)

    # Create model
    model = Model(inputs, x, name=model_name)

    # Load weights
    if weights in ("imagenet", "vggface"):
        wpd = _get_weights_path(weights, include_top)
        weights_path = get_file(wpd["name"], wpd["path"], cache_subdir=wpd["cache_subdir"])
        model.load_weights(weights_path)

    elif weights is not None:
        model.load_weights(weights)

    return model
