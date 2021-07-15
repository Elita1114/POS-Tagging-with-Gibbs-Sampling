from typing import Union, Type
from collections import defaultdict, Counter

Emissions = Type# defaultdict[Union[str, int], Counter]
Transitions = Type #Counter