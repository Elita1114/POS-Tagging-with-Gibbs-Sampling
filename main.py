import argparse
from collections import defaultdict, Counter

from file_utils import *


def normalize_vector(vetor):
    normalize_vector = Counter()
    sum_word = sum(vetor.values())
    for tag in vetor.keys():
        normalize_vector[tag] = vetor[tag] / sum_word
    return normalize_vector


def build_emmission_prob(emission_counter):
    emission_prob = defaultdict(Counter)
    for word in emission_counter.keys():
        emission_prob[word] = normalize_vector(emission_counter[word])
    return emission_prob


def build_emmision(emission_counter, corpus_words, corpus_tags):
    for word, tag in zip(corpus_words, corpus_tags):
        emission_counter[word][tag] += 1
    emission_prob = build_emmission_prob(emission_counter)
    return emission_prob


def build_transition(corpus_tags, transition_counter, transition_length):
    for idx in range(transition_length, len(corpus_tags)):
        transition_counter[tuple(corpus_tags[idx-transition_length:idx+1])]+=1
    transition_prob = normalize_vector(transition_counter)
    return transition_prob

def main():
    corpus_words, corpus_tags, indeces_untagged_words, lemmas = arrange_corpus(train_path, dev_path, test_path, PADDING_START_END_length, pos_tags_learning)
    emission_counter = defaultdict(Counter)
    transition_counter = Counter()
    emission = build_emmision(emission_counter, corpus_words, corpus_tags)
    transition = build_transition(corpus_tags, transition_counter, transition_length)

    i =1








if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description="learning POS tags UNSUPERVISED!!"
    )

    argparser.add_argument("--train_path", type=str, default="data/ud-treebanks-v2.8/ud-treebanks-v2.8/UD_English-GUM/en_gum-ud-train.conllu", help="train_path.")
    argparser.add_argument("--dev_path", type=str, default="data/ud-treebanks-v2.8/ud-treebanks-v2.8/UD_English-GUM/en_gum-ud-dev.conllu", help="dev_path.")
    argparser.add_argument("--test_path", type=str, default="data/ud-treebanks-v2.8/ud-treebanks-v2.8/UD_English-GUM/en_gum-ud-test.conllu", help="test_path.")
    argparser.add_argument("--PADDING_START_END_length", type=int, default=2, help="PADDING_START_END_length")
    argparser.add_argument("--transition_length", type=int, default=1, help="transition_length")

    args = argparser.parse_args()

    train_path = args.train_path
    dev_path = args.dev_path
    test_path = args.test_path
    PADDING_START_END_length = args.PADDING_START_END_length
    transition_length = args.transition_length

    pos_tags_learning = [".", ","]


    main()