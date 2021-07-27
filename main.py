"""
Main file execute it to run the program.
Submitters: Eliya Nakar & Yotam Firer
"""

import argparse
import numpy as np
import random

from typing import List

from file_utils import arrange_corpus
from gibbs_sampling import GibbsSampler, GibbsSamplingArguments
from build_em_trans_utils import build_transitions, build_emissions


def main():
    corpus_words, corpus_tags, indexes_untagged_words, lemmas = arrange_corpus(
        TRAIN_PATH,
        DEV_PATH,
        TEST_PATH,
        PADDING_LENGTH,
        POS_TAGS_WE_ARE_LEARNING
    )

    gold_tags = corpus_tags.copy()

    corpus_tags = np.array(corpus_tags)
    initialize_random_tags = np.array(random.choices(
        range(len(POS_TAGS_WE_ARE_LEARNING)),
        k=len(indexes_untagged_words)
    ))

    corpus_tags[indexes_untagged_words] = initialize_random_tags
    corpus_tags = corpus_tags.tolist()

    emission_counter, emission_probs = build_emissions(corpus_words, corpus_tags)
    # transition_counter, transition_probs = build_transitions(corpus_tags, WINDOW_SIZE)
    transition_counter = build_transitions(corpus_tags, WINDOW_SIZE)

    gibbs_args = GibbsSamplingArguments(
        corpus_words=corpus_words,
        corpus_tags=corpus_tags,
        indexes_of_untagged_words=indexes_untagged_words,
        emission_probs=emission_probs,
        emission_counter=emission_counter,
        transition_counter=transition_counter,
        learning_tags=POS_TAGS_WE_ARE_LEARNING,
        window_size=WINDOW_SIZE,
        gold_tags=gold_tags
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

    # parse the command-line arguments
    args = argparser.parse_args()

    TRAIN_PATH: str = args.train_path
    DEV_PATH: str = args.dev_path
    TEST_PATH: str = args.test_path
    WINDOW_SIZE: int = args.window_size
    PADDING_LENGTH: int = WINDOW_SIZE * 2

    POS_TAGS_WE_ARE_LEARNING: List[str] = [".", ","]

    # call the main function
    main()
