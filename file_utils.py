"""This file contains utility functions related to reading/writing to files."""

from tqdm import tqdm
from constants import START_PAD, END_PAD, field_values_dictionary

from typing import Union, List, Set


def arrange_corpus(train_path: str,
                   dev_path: str,
                   test_path: str,
                   padding: int,
                   pos_tags_we_are_learning: List[str]):
    """
    Read and Arrange the corpus.

    :param train_path: The path to the train data file.
    :param dev_path: The path to the dev data file.
    :param test_path: The path to the test data file.
    :param padding: The padding length to add (has to be >= 0).
    :param pos_tags_we_are_learning: The POS tags we are learning.
    :return: The corpus' words, The corpus' tags, The corpus' indexes of untagged words and The corpus' lemmas.
    """

    # read the train data
    corpus_words_train, corpus_tags_train, indexes_untagged_words_train, lemmas_train \
        = _read_data(train_path, padding=padding, pos_tags_we_are_learning=pos_tags_we_are_learning)

    # read the dev data
    corpus_words_dev, corpus_tags_dev, indexes_untagged_words_dev, lemmas_dev\
        = _read_data(dev_path, padding=padding, pos_tags_we_are_learning=pos_tags_we_are_learning,
                     start_index=len(corpus_tags_train))

    corpus_words = corpus_words_train + corpus_words_dev
    corpus_tags = corpus_tags_train + corpus_tags_dev
    indexes_untagged_words = indexes_untagged_words_train + indexes_untagged_words_dev
    lemmas = lemmas_train.union(lemmas_dev)

    # read the test data
    corpus_words_test, corpus_tags_test, indexes_untagged_words_test, lemmas_test \
        = _read_data(test_path, padding=padding, pos_tags_we_are_learning=pos_tags_we_are_learning,
                     start_index=len(corpus_tags))

    # add the test words, tags, untagged word indexes and lemmas
    # to the corresponding corpus' ones
    corpus_words += corpus_words_test
    corpus_tags += corpus_tags_test
    indexes_untagged_words += indexes_untagged_words_test
    lemmas = lemmas.union(lemmas_test)

    # return the arranged data
    return corpus_words, corpus_tags, indexes_untagged_words, lemmas


def _read_data(
        corpus_file_path: str,
        padding: int,
        pos_tags_we_are_learning: List[str],
        start_index: int = 0
) -> Union[List[str], List[str], List[int], Set[str]]:
    """
       Splits the corpus' words into list of sentences.
       Also collects all the unique lemmas.
       Each sentence is a list of dictionaries.
       Each dictionary represents a word's fields.
       The field names are the keys and the field values are the dictionary values.

       :param corpus_file_path: the path to the corpus_words file.
       :return: The corpus' words split into a list of sentences,
                and the unique lemmas from the corpus' words.
       """

    flag_end_sentence = False
    indexes_untagged_words = list()

    corpus_words: List[str] = list()
    corpus_tags_gold: List[str] = list()
    lemmas: Set[str] = set()

    corpus_words += [START_PAD] * padding
    corpus_tags_gold += [START_PAD] * padding

    with open(corpus_file_path, 'r', encoding='utf-8') as corpus_file:
        # read the file line by line
        # with this reading method we load little components of the corpus_words
        # and we preventing memory errors
        for line in tqdm(corpus_file):
            # delete '\n's in the end of the line
            if line[0] == "" or line[0] == "#":
                continue
            # if the line is a space line between sentences
            # move to the next sentence
            elif line == '\n':
                # append the sentence into the sentences matrix
                # this matrix contains the data about all the sentences
                # each sentence contains the data about its words
                if flag_end_sentence:
                    continue
                flag_end_sentence = True
                corpus_words += ([END_PAD] * padding)
                corpus_words += ([START_PAD] * padding)
                corpus_tags_gold += ([END_PAD] * padding)
                corpus_tags_gold += ([START_PAD] * padding)

            # a normal line containing a word fields
            else:
                # split the line into the filed values
                line = line.strip('\n')
                field_values = line.split('\t')

                corpus_words.append(field_values[field_values_dictionary["LEMMA"]])
                corpus_tags_gold.append(field_values[field_values_dictionary["POSTAG"]])
                lemmas.add(field_values[field_values_dictionary["LEMMA"]])
                flag_end_sentence = False

                if field_values[field_values_dictionary["POSTAG"]] in pos_tags_we_are_learning:
                    indexes_untagged_words.append(len(corpus_tags_gold) - 1 + start_index)

    return corpus_words[:-2], corpus_tags_gold[:-2], indexes_untagged_words, lemmas
