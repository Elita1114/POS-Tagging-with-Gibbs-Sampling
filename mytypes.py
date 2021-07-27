"""This file contains special types we declare."""


from typing import Union
from collections import defaultdict, Counter

Tag = Union[str, int]
Emissions = defaultdict[Tag, Counter]
Transitions = Counter

