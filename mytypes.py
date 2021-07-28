"""This file contains special types we declare."""


from typing import Union, DefaultDict
from collections import Counter

Tag = Union[str, int]
Emissions = DefaultDict[Tag, Counter]
Transitions = Counter

