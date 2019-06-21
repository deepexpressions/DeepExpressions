from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_file
from .applications_utils import _obtain_input_shape


# Keras RESNET50 Implementation
# https://github.com/keras-team/keras-applications/
RESNET50_WEIGHTS_PATH = dict(
    name = "resnet50_weights_tf_dim_ordering_tf_kernels.h5",
    path = ("https://github.com/fchollet/deep-learning-models/releases/"
            "download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5"),
    cache_subdir = "models",
)
RESNET50_WEIGHTS_PATH_NO_TOP = dict(
    name = "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",
    path = ("https://github.com/fchollet/deep-learning-models/releases/"
            "download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5"),
    cache_subdir = "models",
)


# Oxford VGGFace Implementation using Keras Functional Framework v2+
# https://github.com/rcmalli/keras-vggface
VGG_FACE_WEIGHTS_PATH = dict(
    name = "rcmalli_vggface_tf_resnet50.h5",
    path = ("https://github.com/rcmalli/keras-vggface/releases/"
            "download/v2.0/rcmalli_vggface_tf_resnet50.h5"),
    cache_subdir = "models/vgg_face",
)
VGG_FACE_WEIGHTS_PATH_NO_TOP = dict(
    name = "rcmalli_vggface_tf_notop_resnet50.h5",
    path = ("https://github.com/rcmalli/keras-vggface/releases/"
            "download/v2.0/rcmalli_vggface_tf_notop_resnet50.h5"),
    cache_subdir = "models/vgg_face",
)


def _get_weights_path(weights, include_top):
    if weights == "vggface":
        weights_path = VGG_FACE_WEIGHTS_PATH if include_top else VGG_FACE_WEIGHTS_PATH_NO_TOP
    
    elif weights == "imagenet":
        weights_path = RESNET50_WEIGHTS_PATH if include_top else RESNET50_WEIGHTS_PATH_NO_TOP

    return weights_path


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1),
               kernel_initializer='he_normal',
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same',
               kernel_initializer='he_normal',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1),
               kernel_initializer='he_normal',
               name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def RESNET50(include_top=False, weights=None, input_shape=(224, 224, 3),
          pooling=None, num_classes=1000, model_name="resnet50"):


    if not (weights in {"imagenet", "vggface", None} or os.path.exists(weights)):
        raise ValueError("The `weights` argument should be either "
                         "`None` (random initialization), `imagenet`, `vggface` "
                         "or the path to the weights file to be loaded.")

    if weights == "imagenet" and include_top and classes != 1000:
        raise ValueError("If using `weights` as `imagenet` with `include_top` "
                         "as true, `classes` should be 1000")

    if weights == "vggface" and include_top and classes != 8631:
        raise ValueError("If using `weights` as `imagenet` with `include_top` "
                         "as true, `classes` should be 8631")


    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    # Model input
    inputs = Input(shape=input_shape)

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(inputs)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    if include_top:
        # Classification Block
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(num_classes, activation='softmax', name='fc1000')(x)

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
