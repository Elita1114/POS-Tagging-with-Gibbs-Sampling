"""
This file contains utility functions related to building the
emission & transition, counters & probability vectors.
"""

from mytypes import Emissions, Transitions
from collections import defaultdict
from non_negative_counter import NonNegativeCounter
from typing import Iterable, Tuple
from math_utils import normalize


def get_emission_probs(emission_counter: Emissions) -> Emissions:
    """Get the emission probabilities by normalizing the given emission counter."""
    emission_prob = defaultdict(NonNegativeCounter)

    for word in emission_counter.keys():
        emission_prob[word] = normalize(emission_counter[word])

    return emission_prob


def build_emissions(corpus_words: Iterable, corpus_tags: Iterable) -> Tuple[Emissions, Emissions]:
    """Calculate and Build the emission probabilities."""
    emission_counter = defaultdict(NonNegativeCounter)

    for word, tag in zip(corpus_words, corpus_tags):
        emission_counter[word][tag] += 1

    emission_prob = get_emission_probs(emission_counter)
    return emission_counter, emission_prob


def build_transitions(corpus_tags, transition_length) -> Transitions:
    """Calculate and Build the transition probabilities"""
    transition_counter = NonNegativeCounter()

    for idx in range(transition_length, len(corpus_tags)):
        transition_counter[
            tuple(
                corpus_tags[idx - transition_length: idx + 1]
            )
        ] += 1
        transition_counter[corpus_tags[idx]] += 1

    return transition_counter
