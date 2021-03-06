"""This file contains special types we declare."""


from typing import Union, DefaultDict
from non_negative_counter import NonNegativeCounter

Tag = Union[str, int]
AbstractTag = Tag
Emissions = DefaultDict[Tag, NonNegativeCounter]
Transitions = NonNegativeCounter

