"""
Facial Expressions Recognition Classifier

Model based on VGG FACE implemented by Refik Can Malli 
(https://github.com/rcmalli/keras-vggface).

Training using a subset from the Compound Emotions Dataset
(https://www.pnas.org/content/111/15/E1454).
"""


from .applications import VGG16

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_file


VGGFACE_CE_WEIGHTS_PATH = dict(
    name = None,
    path = (None),
    cache_subdir = "models/deep_expressions",
)


def VGGFACE_CE(input_shape=(224, 224, 3), model_name="vggface_ce"):

    vggface = VGG16(input_shape=input_shape, model_name=model_name)

    # Classification Block
    x = vggface.get_layer("block4_pool").output
    x = Flatten(name="flatten")(x)
    x = Dense(256, activation="relu", name="fc1")(x)
    x = Dense(128, activation="relu", name="fc2")(x)
    x = Dense(7, activation="softmax", name="predictions")(x)

    # Create model
    model = Model(vggface.input, x, name=model_name)

    # Load weights
    weights_path = get_file(
        VGGFACE_CE_WEIGHTS_PATH["name"], 
        VGGFACE_CE_WEIGHTS_PATH["path"], 
        cache_subdir=VGGFACE_CE_WEIGHTS_PATH["cache_subdir"])
        
    model.load_weights(weights_path)

    return model


    