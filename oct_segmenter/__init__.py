import os

import configparser
from pathlib import Path
from prettytable import PrettyTable

__appname__ = "octsegmenter"

# Semantic Versioning 2.0.0: https://semver.org/
# 1. MAJOR version when you make incompatible API changes;
# 2. MINOR version when you add functionality in a backwards-compatible manner;
# 3. PATCH version when you make backwards-compatible bug fixes.
__version__ = "0.6.0"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

MODELS_DIR = Path(os.path.dirname(os.path.abspath(__file__)) + "/data/models/")
CONFIG_FILE_PATH = Path.home() / Path(".oct-segmenter/config")


def load_models_table():
    """
    The conventions in this functions are:
    1. The name of the model shown is the name of its relative path starting from data/model
    2. Each directory must contain at most 1 model
    3. The model name must start with "model*" and end with "*.hdf5"
    """

    models = {}
    for subdir, dirs, files in os.walk(MODELS_DIR):
        for file in files:
            if file.startswith("model") and file.endswith(".hdf5"):
                models[str(Path(subdir).relative_to(MODELS_DIR))] = Path(os.path.join(subdir, file))

    models_ascii = PrettyTable() # Build models ascii table for listing
    models_ascii.field_names = ["Default", "Selection", "Model Name"]
    default_model_index = None
    models_index_map = {} # Maps index -> model names
    for i, model_name in enumerate(models.keys()):
        models_index_map[i] = model_name
        if DEFAULT_MODEL_NAME == model_name:
            default_model_index = i
            models_ascii.add_row(["*", i, model_name])
        else:
            models_ascii.add_row(["", i, model_name])

    return models, models_ascii, models_index_map, default_model_index


def write_default_config():
    if not CONFIG_FILE_PATH.is_file():
        config = configparser.ConfigParser()
        config["DEFAULT"] = { "model_dir": "visual-function-core"}
        config["User"] = {}

        os.makedirs(CONFIG_FILE_PATH.parent, exist_ok=True)
        with open(CONFIG_FILE_PATH, "w") as config_file:
            config.write(config_file)


def load_config() -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE_PATH)
    return config


write_default_config()
CONFIG = load_config()
DEFAULT_MODEL_NAME = CONFIG.get("User", "model_dir")
MODELS_TABLE, MODELS_TABLE_ASCII, MODELS_INDEX_MAP, DEFAULT_MODEL_INDEX = load_models_table()

DEFAULT_TEST_PARTITION = 0.3
DEFAULT_TRAINING_PARTITION = round(0.8 * (1 - DEFAULT_TEST_PARTITION), 2)
DEFAULT_VALIDATION_PARTITION = round(0.2 * (1 - DEFAULT_TEST_PARTITION), 2)
