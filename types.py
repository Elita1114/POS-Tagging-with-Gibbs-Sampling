from typing import Union
from collections import defaultdict, Counter

Emissions = defaultdict[Union[str, int], Counter]
Transitions = Counter
