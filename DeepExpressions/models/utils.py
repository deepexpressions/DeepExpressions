import os
import json
from tensorflow.keras.utils import get_file


FILE_PATH = dict(
    name = "models.json",
    path = "https://raw.githubusercontent.com/deepexpressions/models/master/models.json",
    cache_subdir = "models/deep_expressions/",
)

# FILENAME = os.path.join(os.getenv("HOME"), ".keras/models/deep_expressions/models.json")
FILENAME = "./models.json"


def list_models(only_names=True):

    # Download models.json file
    get_file(FILE_PATH["name"], 
             FILE_PATH["path"],
             cache_subdir=FILE_PATH["cache_subdir"])
    
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
        

def get_model_labels(model_name):

    # Download models.json file
    get_file(FILE_PATH["name"], 
             FILE_PATH["path"],
             cache_subdir=FILE_PATH["cache_subdir"])
    
    # Load models.json
    with open(FILENAME, "r") as f:
        data = json.load(f)

    if model_name not in data.keys():
        raise ValueError("Unknown model.")

    # Return model labels
    labels = data[model_name]["labels"]
    return {int(k) : v for k, v in labels.items()}