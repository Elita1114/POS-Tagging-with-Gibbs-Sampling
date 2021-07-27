"""This file contains implementations of math operations."""

import numpy as np
from collections import defaultdict, Counter

from typing import Union

_Vector_Types = Union[list, np.array, dict, defaultdict, Counter]


def normalize(vector: _Vector_Types) -> _Vector_Types:
    """Return a normalized version of the given vector"""
    if type(vector) == list:
        sum_of_values = sum(vector)
        return [float(i) / sum_of_values for i in vector]

    elif type(vector) == np.array:
        return np.linalg.norm(vector)

    elif type(vector) in {dict, defaultdict, Counter}:
        sum_of_values = float(sum(vector.values()))
        cls = type(vector)

        return cls({
                tag: vector[tag] / sum_of_values for tag in vector.keys()
            })
    else:
        raise AttributeError(f"Given vectors' type ({type(vector)}) is not supported.")
