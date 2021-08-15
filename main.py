"""
Main file. execute it to run the program.
Submitters: Eliya Nakar & Yotam Firer.

Bar Ilan University, Department of Computer Science.
"""

import argparse
import numpy as np
import random

from typing import List

from general_utils import set_seed
from mytypes import Tag

from file_utils import arrange_corpus
from gibbs_sampling import GibbsSampler, GibbsSamplingArguments
from build_em_trans_utils import build_transitions, build_emissions
from plotit import plot_scores


def main():
    set_seed(seed=SEED)

    # read the data and parse it
    corpus_words, corpus_tags, indexes_of_untagged_words, lemmas = arrange_corpus(
        train_path=TRAIN_PATH,
        dev_path=DEV_PATH,
        test_path=TEST_PATH,
        padding=PADDING_LENGTH,
        pos_tags_we_are_learning=POS_TAGS_WE_ARE_LEARNING
    )

    gold_tags = corpus_tags.copy()

    corpus_tags = _set_random_tags(corpus_tags, indexes_of_untagged_words)

    emission_counter, emission_probs = build_emissions(corpus_words, corpus_tags)
    # transition_counter, transition_probs = build_transitions(corpus_tags, WINDOW_SIZE)
    transition_counter = build_transitions(corpus_tags, WINDOW_SIZE)

    gibbs_args = GibbsSamplingArguments(
        corpus_words=corpus_words,
        corpus_tags=corpus_tags,
        indexes_of_untagged_words=indexes_of_untagged_words,
        emission_probs=emission_probs,
        emission_counter=emission_counter,
        transition_counter=transition_counter,
        learning_tags=POS_TAGS_WE_ARE_LEARNING,
        abstract_tags=range(len(POS_TAGS_WE_ARE_LEARNING)),
        window_size=WINDOW_SIZE,
        gold_tags=gold_tags
    )

    gibbs_sampler = GibbsSampler(gibbs_args)

    results, scores = gibbs_sampler.run(epochs_number=EPOCHS, save_at_end=SAVE_AT_END, verbose=VERBOSE)
    plot_scores(scores)


def _set_random_tags(tag_list: List[Tag], indexes_to_set_to: List[int]) -> List[Tag]:
    """Return the passed tag list with random tags in the given indexes list."""
    tag_list = np.array(tag_list)

    random_tags = np.array(random.choices(
        range(len(POS_TAGS_WE_ARE_LEARNING)),
        k=len(indexes_to_set_to)
    ))

    tag_list[indexes_to_set_to] = random_tags
    tag_list = tag_list.tolist()

    return tag_list


def _validate_the_args():
    """
    This function raises an exception if one of the global arguments is not valid.
    (Used only once after parsing the arguments).
    """
    assert TRAIN_PATH is not None, 'The training set path is None, which is invalid.'
    assert DEV_PATH is not None, 'The dev set path is None, which is invalid.'
    assert TEST_PATH is not None, 'The test set path is None, which is invalid.'

    assert WINDOW_SIZE >= 0, 'Window size has to be a non-negative number.'


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description="Learning POS tags UNSUPERVISED!!"
    )

    argparser.add_argument(
        "--train-path",
        type=str,
        default="en_gum-ud-train.conllu",
        help="Train file path."
    )

    argparser.add_argument(
        "--dev-path",
        type=str,
        default="en_gum-ud-dev.conllu",
        help="Dev file path."
    )

    argparser.add_argument(
        "--test-path",
        type=str,
        default="en_gum-ud-test.conllu",
        help="Test file path."
    )

    argparser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="The random seed to use."
    )

    argparser.add_argument(
        "--window-size",
        type=int,
        default=1,
        help="The tags window size."
    )

    argparser.add_argument(
        "--epochs", '-e',
        type=int,
        default=1000,
        help="The epochs to run for."
    )

    argparser.add_argument(
        "--save-at-end",
        action='store_true',
        help="If set the arguments will be saved at the end."
    )

    argparser.add_argument(
        "--silent",
        action='store_true',
        help="If set the run will not output the scores for each epoch."
    )

    # parse the command-line arguments
    args = argparser.parse_args()

    TRAIN_PATH: str = args.train_path
    DEV_PATH: str = args.dev_path
    TEST_PATH: str = args.test_path

    SEED: int = args.seed

    EPOCHS: int = args.epochs
    SAVE_AT_END: bool = args.save_at_end

    WINDOW_SIZE: int = args.window_size
    PADDING_LENGTH: int = WINDOW_SIZE * 2
    VERBOSE: bool = not args.silent

    _validate_the_args()

    POS_TAGS_WE_ARE_LEARNING: List[str] = [ '-LRB-', '-RRB-']

    # call the main function
    main()
