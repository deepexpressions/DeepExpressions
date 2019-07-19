import os
import json
import warnings
from tensorflow.keras.utils import get_file

from . import vggface
# from . import inception (to be implemented)


FILENAME = os.path.join(os.getenv("HOME"), ".keras/models/deep_expressions/models.json")


def _get_models_json_file():
    """Download models.json file from DeepExpressions models repo."""

    get_file("models.json", 
             "https://raw.githubusercontent.com/deepexpressions/models/master/models.json",
             cache_subdir = "models/deep_expressions/")


def _model_exists(name):
    """Check if a model exists in DeepExpressions models."""

    # Download models.json file
    _get_models_json_file()

    # Load models.json
    with open(FILENAME, "r") as f:
        data = json.load(f)

    return True if name in data.keys() else False

    
def _get_model_labels(name):
    """Get labels given a model name."""

    # Check if model_name exists
    if not _model_exists(name):
        raise Exception("`{}` is not a valid name for a model.".format(name))
    
    # Load models.json
    with open(FILENAME, "r") as f:
        data = json.load(f)

    # Return model labels
    labels = data[name]["labels"]
    return {int(k) : v for k, v in labels.items()}


def print_model_info(name):
    """Prints DeepExpressions model informations."""

    # Check if model_name exists
    if not _model_exists(name):
        raise Exception("`{}` is not a valid name for a model.".format(name))
    
    # Load models.json
    with open(FILENAME, "r") as f:
        data = json.load(f)


    # Print model info
    print("DeepExpressions models:")
    print("-----------------------")
    print(" * ", name)
    print("    ConvNet: ", data[name]["convnet"])
    print("    Database: ", data[name]["database"])
    print("    Derivations: ", data[name]["derivations"])
    print("    Labels: ", list(data[name]["labels"].values()), "\n")


def list_models(only_names=True):
    """List DeepExpressions trained models informations."""

    # Download models.json file
    _get_models_json_file()
    
    # Load models.json
    with open(FILENAME, "r") as f:
        data = json.load(f)

    # Print models name
    if only_names:
        print("DeepExpressions models:")
        print("-----------------------")
        [print(" * ", name) for name in data]

    # Print models info
    else:
        print("DeepExpressions models:")
        print("-----------------------")

        for name in data:
            print(" * ", name)
            print("    ConvNet: ", data[name]["convnet"])
            print("    Database: ", data[name]["database"])
            print("    Derivations: ", data[name]["derivations"])
            print("    Labels: ", list(data[name]["labels"].values()), "\n")


def load_model(name, input_shape):
    """
    Load a Facial Expression Recognition classifier from DeepExpressions models.

    Arguments:
        + name (str) -- Name of the model. Options available at : ?.

    Output:
        + Keras model for Facial Expression Recognition.
        + Preprocess input function
    """

    print("[INFO] Loading model '{}' ...".format(name), end=" ")

    if name in {"vggface_ce", "vggface_ce_l1"}:
        model = vggface.VGGFACE(weights="ce", num_layers=1 , num_classes=7, input_shape=input_shape)

    elif name == "vggface_ce_l3":
        model = vggface.VGGFACE(weights="ce", num_layers=3 , num_classes=7, input_shape=input_shape)
    
    elif name in {"vggface_ck", "vggface_ck+", "vggface_ck_l1", "vggface_ck+_l1"}:
        model = vggface.VGGFACE(weights="ck", num_layers=1 , num_classes=7, input_shape=input_shape)

    elif name in {"vggface_ck_l3", "vggface_ck+_l3"}:
        model = vggface.VGGFACE(weights="ck", num_layers=3 , num_classes=7, input_shape=input_shape)

    elif name in {"vggface_kdef", "vggface_kdef_l1"}:
        model = vggface.VGGFACE(weights="kdef", num_layers=1 , num_classes=7, input_shape=input_shape)

    elif name == "vggface_kdef_l3":
        model = vggface.VGGFACE(weights="kdef", num_layers=3 , num_classes=7, input_shape=input_shape)

    elif name == "vggface_iffg":
        model = vggface.VGGFACE(weights="iffg", num_layers=3, num_classes=9, input_shape=input_shape)

    else:
        raise Exception("Unknown model.")

    print("Done.")

    return model


def load_model_from_dir(filename):
    """
    Load a Facial Expression Recognition classifier from DeepExpressions models.

    Arguments:
        + filename (str) -- Path to custom model saved (only *.h5 files).

    Output:
        + Keras model for Facial Expression Recognition.
    """

    # Verify file extension
    assert filename.endswith(".h5"), "Custom model must be save with extenstion '.h5'."

    print("[INFO] Loading model '{}' ...".format(name), end=" ")
    model = tf.keras.models.load_model(filename)
    print("Done.")

    return model    


def _obtain_input_shape(input_shape,
                        default_size,
                        min_size,
                        data_format,
                        require_flatten,
                        weights=None):
    """Internal utility to compute/validate a model's input shape.
    # Arguments
        input_shape: Either None (will return the default network input shape),
            or a user-provided shape to be validated.
        default_size: Default input width/height for the model.
        min_size: Minimum input width/height accepted by the model.
        data_format: Image data format to use.
        require_flatten: Whether the model is expected to
            be linked to a classifier via a Flatten layer.
        weights: One of `None` (random initialization)
            or 'imagenet' (pre-training on ImageNet).
            If weights='imagenet' input channels must be equal to 3.
    # Returns
        An integer shape tuple (may include None entries).
    # Raises
        ValueError: In case of invalid argument values.
    """
    if weights != 'imagenet' and input_shape and len(input_shape) == 3:
        if data_format == 'channels_first':
            if input_shape[0] not in {1, 3}:
                warnings.warn(
                    'This model usually expects 1 or 3 input channels. '
                    'However, it was passed an input_shape with ' +
                    str(input_shape[0]) + ' input channels.')
            default_shape = (input_shape[0], default_size, default_size)
        else:
            if input_shape[-1] not in {1, 3}:
                warnings.warn(
                    'This model usually expects 1 or 3 input channels. '
                    'However, it was passed an input_shape with ' +
                    str(input_shape[-1]) + ' input channels.')
            default_shape = (default_size, default_size, input_shape[-1])
    else:
        if data_format == 'channels_first':
            default_shape = (3, default_size, default_size)
        else:
            default_shape = (default_size, default_size, 3)
    if weights == 'imagenet' and require_flatten:
        if input_shape is not None:
            if input_shape != default_shape:
                raise ValueError('When setting `include_top=True` '
                                 'and loading `imagenet` weights, '
                                 '`input_shape` should be ' +
                                 str(default_shape) + '.')
        return default_shape
    if input_shape:
        if data_format == 'channels_first':
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError(
                        '`input_shape` must be a tuple of three integers.')
                if input_shape[0] != 3 and weights == 'imagenet':
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape=' + str(input_shape) + '`')
                if ((input_shape[1] is not None and input_shape[1] < min_size) or
                   (input_shape[2] is not None and input_shape[2] < min_size)):
                    raise ValueError('Input size must be at least ' +
                                     str(min_size) + 'x' + str(min_size) +
                                     '; got `input_shape=' +
                                     str(input_shape) + '`')
        else:
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError(
                        '`input_shape` must be a tuple of three integers.')
                if input_shape[-1] != 3 and weights == 'imagenet':
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape=' + str(input_shape) + '`')
                if ((input_shape[0] is not None and input_shape[0] < min_size) or
                   (input_shape[1] is not None and input_shape[1] < min_size)):
                    raise ValueError('Input size must be at least ' +
                                     str(min_size) + 'x' + str(min_size) +
                                     '; got `input_shape=' +
                                     str(input_shape) + '`')
    else:
        if require_flatten:
            input_shape = default_shape
        else:
            if data_format == 'channels_first':
                input_shape = (3, None, None)
            else:
                input_shape = (None, None, 3)
    if require_flatten:
        if None in input_shape:
            raise ValueError('If `include_top` is True, '
                             'you should specify a static `input_shape`. '
                             'Got `input_shape=' + str(input_shape) + '`')
    return input_shape