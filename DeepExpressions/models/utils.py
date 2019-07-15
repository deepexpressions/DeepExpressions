import os
import json
from tensorflow.keras.utils import get_file


FILE_PATH = dict(
    name = "models.json",
    path = "https://raw.githubusercontent.com/deepexpressions/models/master/models.json",
    cache_subdir = "models/deep_expressions/",
)

FILENAME = os.path.join(os.getenv("HOME"), ".keras/models/deep_expressions/models.json")


def _get_models_json_file():
    """Download models.json file from DeepExpressions models repo."""

    get_file(FILE_PATH["name"], 
             FILE_PATH["path"],
             cache_subdir=FILE_PATH["cache_subdir"])


def _get_model_url(name):
    
    # Download models.json file
    _get_models_json_file()
    
    # Load models.json
    with open(FILENAME, "r") as f:
        data = json.load(f)

    return data[name]["url"]

    
def _get_model_labels(name):
    """Get labels given a model name."""

    # Download models.json file
    _get_models_json_file()
    
    # Load models.json
    with open(FILENAME, "r") as f:
        data = json.load(f)

    if name not in data.keys():
        raise Exception("Unknown model.")

    # Return model labels
    labels = data[name]["labels"]
    return {int(k) : v for k, v in labels.items()}


def _model_exists(name):
    """Check if a model exists in DeepExpressions models."""

    # Download models.json file
    _get_models_json_file()

    # Load models.json
    with open(FILENAME, "r") as f:
        data = json.load(f)

    return True if name in data.keys() else False


# def load_model(name):
#     """
#     Load a Facial Expression Recognition classifier from DeepExpressions models.

#     Arguments:
#         + name (str) -- Name of the model. Options available at : ?.
#         + filename (str) -- Path to custom model saved (only *.h5 files).

#     Output:
#         + Keras model for Facial Expression Recognition.
#     """
#     def maybe_download_model(name):
#         # Use keras utils to download model
#         model_path = get_file(
#             name + ".h5",
#             _get_model_url(name),
#             cache_subdir="models/deep_expressions/"
#         )
#         return model_path

#     print("[INFO] Loading model '{}' ...".format(name), end=" ")

#     model_path = maybe_download_model(name)

#     # Load model
#     model = tf.keras.models.load_model(model_path)
#     print("Done.")

#     return model


# def load_model_from_dir(filename):
#     """
#     Load a Facial Expression Recognition classifier from DeepExpressions models.

#     Arguments:
#         + name (str) -- Name of the model. Options available at : ?.
#         + filename (str) -- Path to custom model saved (only *.h5 files).

#     Output:
#         + Keras model for Facial Expression Recognition.
#     """

#     # Verify file extension
#     assert filename.endswith(".h5"), "Custom model must be save with extenstion '.h5'."

#     print("[INFO] Loading model '{}' ...".format(name), end=" ")
#     model = tf.keras.models.load_model(filename)
#     print("Done.")

#     return model    


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
            print("    Labels: ", list(data[name]["labels"].values()), "\n")
