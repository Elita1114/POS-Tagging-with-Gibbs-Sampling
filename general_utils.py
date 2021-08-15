"""This file contains general utility (helper) functions"""

import pickle
import random
import numpy as np


def set_seed(seed: int) -> None:
    """Set the random seed"""
    random.seed(seed)
    np.random.seed(seed)


def save_object(obj: object, filename: str) -> None:
    """Save an object in file"""
    with open(filename, 'wb') as output:
        pickle.dump(obj, output)


def load_object(filename: str) -> object:
    """Load an object saved with *save_object*"""
    with open(filename, 'rb') as file:
        return pickle.load(file)