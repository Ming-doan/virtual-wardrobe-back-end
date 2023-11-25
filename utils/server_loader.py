# Utilities for read and write server data file

import json
import os
import numpy as np

DIRECTORY = 'model'
CURRENT_PATH = os.getcwd()


def load_json(path):
    with open(os.path.join(CURRENT_PATH, DIRECTORY, path), 'r') as f:
        return json.load(f)


def write_json(path, data: dict):
    with open(os.path.join(CURRENT_PATH, DIRECTORY, path), 'w') as f:
        json.dump(data, f, indent=4)


def read_numpy(path):
    return np.load(os.path.join(CURRENT_PATH, DIRECTORY, path))


def write_numpy(path, data: np.ndarray):
    np.save(os.path.join(CURRENT_PATH, DIRECTORY, path), data)
