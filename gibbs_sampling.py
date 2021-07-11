from dataclasses import dataclass
from typing import Union, Iterable
from types import Emissions, Transitions
from numpy.random import choice

from math_utils import normalize



@dataclass(eq=False)
class GibbsSamplingArguments(object):
    corpus_words: Iterable
    corpus_tags: Iterable
    indexes_untagged_words: Iterable
    emission_probs: Emissions
    emission_counter: Emissions
    transition_probs: Transitions
    transition_counter: Transitions
    learning_tags: Iterable


class GibbsSampler(object):
    def __init__(self, args: GibbsSamplingArguments):
        self.args = args

    def run(self) -> Iterable[Union[int, str]]:
        """Run the gibbs sampling algorithm"""
        for idx in self.args.indexes_untagged_words:
            tag_probs = list()  # replace later with calc_prob()
            tag_probs = [float(i) / sum(tag_probs) for i in tag_probs]
            self.args.corpus_tags[idx] = choice(range(len(tag_probs)), 1, p=tag_probs)[0]


    def calc_prob(self):
