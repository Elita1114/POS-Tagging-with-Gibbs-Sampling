import numpy as _np
from collections import Counter as _Counter


class NonNegativeCounter(_Counter):
    """This class is a Counter sub-class that only accepts non-negative integers as values."""
    def __setitem__(self, key, value):
        if type(value) not in {int, _np.int}:
            raise TypeError("The value to be set is not an integer.")

        if value >= 0:
            super().__setitem__(key, value)
        else:
            raise ValueError(f"Tried setting a negative value ({value}) in key '{key}'")
