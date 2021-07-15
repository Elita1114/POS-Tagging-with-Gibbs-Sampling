from mytypes import Emissions, Transitions
from collections import defaultdict, Counter
from typing import Union, Iterable, Tuple
from math_utils import normalize



def normalize_emissions(emission_counter: Emissions) -> Emissions:
    """Normalize the emission probabilities from the given emission counter"""
    emission_prob = defaultdict(Counter)

    for word in emission_counter.keys():
        emission_prob[word] = normalize(emission_counter[word])

    return emission_prob


def build_emissions(corpus_words: Iterable, corpus_tags: Iterable) -> Tuple[Emissions, Emissions]:
    """Calculate and Build the emission probabilities"""
    emission_counter = defaultdict(Counter)

    for word, tag in zip(corpus_words, corpus_tags):
        emission_counter[word][tag] += 1

    emission_prob = normalize_emissions(emission_counter)
    return emission_counter, emission_prob


def build_transitions(corpus_tags, transition_length) -> Tuple[Transitions, Transitions]:
    """Calculate and Build the emission probabilities"""
    transition_counter = Counter()

    for idx in range(transition_length, len(corpus_tags)):
        transition_counter[
            tuple(
                corpus_tags[idx - transition_length: idx + 1]
            )
        ] += 1
        transition_counter[corpus_tags[idx]] += 1

    #transition_prob = normalize(transition_counter)
    return transition_counter #, transition_prob
