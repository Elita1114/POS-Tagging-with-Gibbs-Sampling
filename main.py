import argparse
from collections import defaultdict, Counter
from typing import Union, Iterable, Tuple

import numpy as np
import random

from file_utils import *
from gibbs_sampling import GibbsSampler, GibbsSamplingArguments
from types import Emissions, Transitions


def normalize_vector(vector: Union[Counter, dict]) -> Counter:
    """Normalize a vector represented as a python dictionary."""
    values_sum = sum(vector.values())

    normalized_vector = Counter({
        vector[tag] / values_sum for tag in vector.keys()
    })

    return normalized_vector


def normalize_emissions(emission_counter: Emissions) -> Emissions:
    """Normalize the emission probabilities from the given emission counter"""
    emission_prob = defaultdict(Counter)

    for word in emission_counter.keys():
        emission_prob[word] = normalize_vector(emission_counter[word])

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

    transition_prob = normalize_vector(transition_counter)
    return transition_counter, transition_prob


def main():
    corpus_words, corpus_tags, indexes_untagged_words, lemmas = arrange_corpus(
        TRAIN_PATH,
        DEV_PATH,
        TEST_PATH,
        PADDING_LENGTH,
        POS_TAGS_WE_ARE_LEARNING
    )

    emission_counter, emission_probs = build_emissions(corpus_words, corpus_tags)
    transition_counter, transition_probs = build_transitions(corpus_tags, WINDOW_SIZE)

    corpus_tags = np.array(corpus_tags)
    initialize_random_tags = np.array(random.choices(
        range(len(POS_TAGS_WE_ARE_LEARNING)),
        k=len(indexes_untagged_words)
    ))

    corpus_tags[indexes_untagged_words] = initialize_random_tags
    corpus_tags = corpus_tags.tolist()

    gibbs_args = GibbsSamplingArguments(
        corpus_words=corpus_words,
        corpus_tags=corpus_tags,
        indexes_untagged_words=indexes_untagged_words,
        emission_probs=emission_probs,
        emission_counter=emission_counter,
        transition_probs=transition_probs,
        transition_counter=transition_counter,
        learning_tags=POS_TAGS_WE_ARE_LEARNING
    )

    gibbs_sampler = GibbsSampler(gibbs_args)

    results = gibbs_sampler.run()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description="Learning POS tags UNSUPERVISED!!"
    )

    argparser.add_argument(
        "--train-path",
        type=str,
        default="data/ud-treebanks-v2.8/ud-treebanks-v2.8/UD_English-GUM/en_gum-ud-train.conllu",
        help="Train file path."
    )

    argparser.add_argument(
        "--dev-path",
        type=str,
        default="data/ud-treebanks-v2.8/ud-treebanks-v2.8/UD_English-GUM/en_gum-ud-dev.conllu",
        help="Dev file path."
    )

    argparser.add_argument(
        "--test-path",
        type=str,
        default="data/ud-treebanks-v2.8/ud-treebanks-v2.8/UD_English-GUM/en_gum-ud-test.conllu",
        help="Test file path."
    )

    argparser.add_argument(
        "--window-size",
        type=int,
        default=1,
        help="The tags window size."
    )

    args = argparser.parse_args()

    TRAIN_PATH: str = args.train_path
    DEV_PATH: str = args.dev_path
    TEST_PATH: str = args.test_path
    WINDOW_SIZE: int = args.window_size
    PADDING_LENGTH: int = WINDOW_SIZE * 2

    POS_TAGS_WE_ARE_LEARNING = [".", ","]

    # call main function
    main()
