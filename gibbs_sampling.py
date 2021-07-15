from dataclasses import dataclass
from typing import Union, Iterable
from mytypes import Emissions, Transitions
from numpy.random import choice
from collections import defaultdict, Counter
from math_utils import normalize
from tqdm import tqdm

@dataclass(eq=False)
class GibbsSamplingArguments(object):
    corpus_words: Iterable
    corpus_tags: Iterable
    indexes_untagged_words: Iterable
    emission_probs: Emissions
    emission_counter: Emissions
    #transition_probs: Transitions
    transition_counter: Transitions
    learning_tags: Iterable
    window_size : int
    gold : Iterable


class GibbsSampler(object):
    def __init__(self, args: GibbsSamplingArguments):
        self.args = args

    def run(self) -> Iterable[Union[int, str]]:
        """Run the gibbs sampling algorithm"""
        for _ in tqdm(range(1000)):
            for idx in self.args.indexes_untagged_words:
                word = self.args.corpus_words[idx]
                prev_tag = self.args.corpus_tags[idx]
                tag_probs = self.calc_prob(idx, word, prev_tag)
                tag_probs = [float(i) / sum(tag_probs) for i in tag_probs]
                new_tag = str(choice(range(len(tag_probs)), 1, p=tag_probs)[0])
                self.args.corpus_tags[idx] = new_tag
                self.update_emmission(word, prev_tag, new_tag, change_emission_probs=True)
                self.update_transition(prev_tag, new_tag, idx)

        tag_to_learned = self.match_tag_to_learned_tag()

    def calc_emmission(self, tag, word):
        """
        calc transition for tag
        :param tag:
        :param word:
        :return:
        """
        emmission_probs_local= normalize(self.args.emission_counter[word])
        return emmission_probs_local[tag]

    def update_emmission(self, word, prev_tag, new_tag, change_emission_probs = False):
        """
        change counts of emmission with the new tag
        :param word:
        :param prev_tag:
        :param new_tag:
        :param change_emission_probs:
        :return:
        """
        self.args.emission_counter[word][prev_tag] -= 1
        self.args.emission_counter[word][new_tag] += 1
        if change_emission_probs:
            self.args.emission_probs[word] = normalize(self.args.emission_counter[word])
            if self.args.emission_counter[word][prev_tag] == 0:
                pass #del self.args.emission_counter[word][prev_tag]

    def reverse_emmission(self, word, prev_tag, new_tag):
        """
        when called after update_emmission (with change_emission_probs = False), then it cancelles all his changes
        :param word:
        :param prev_tag:
        :param new_tag:
        :return:
        """
        self.args.emission_counter[word][prev_tag] += 1
        self.args.emission_counter[word][new_tag] -= 1

    def update_transition(self, prev_tag, new_tag, idx):
        """
        change counts of transition with the new tag
        :param prev_tag:
        :param new_tag:
        :param idx:
        :return:
        """
        self.args.transition_counter[prev_tag] -= 1
        for shift in range(self.args.window_size+2):
            self.args.transition_counter[tuple(self.args.corpus_tags[shift + idx - self.args.window_size: shift + idx + 1])] -= 1

        self.args.transition_counter[new_tag] += 1
        self.args.corpus_tags[idx] = new_tag
        for shift in range(self.args.window_size+2):
            self.args.transition_counter[tuple(self.args.corpus_tags[shift + idx - self.args.window_size: shift + idx + 1])] += 1
        self.args.corpus_tags[idx] = prev_tag

    def reverse_transition(self, prev_tag, new_tag, idx):
        """
        when called after update_transition, then it cancelles al his changes
        :param prev_tag:
        :param new_tag:
        :param idx:
        :return:
        """
        self.args.transition_counter[prev_tag] += 1
        for shift in range(self.args.window_size+2):
            self.args.transition_counter[tuple(self.args.corpus_tags[shift + idx - self.args.window_size: shift + idx + 1])] += 1

        self.args.transition_counter[new_tag] -= 1
        self.args.corpus_tags[idx] = new_tag
        for shift in range(self.args.window_size+2):
            self.args.transition_counter[tuple(self.args.corpus_tags[shift + idx - self.args.window_size: shift + idx + 1])] -= 1
        self.args.corpus_tags[idx] = prev_tag

    def calc_transition(self, idx, new_tag):
        """
        calc transition
        :param idx:
        :param new_tag:
        :return:
        """
        transition = 1
        for shift in range(self.args.window_size+2):
            transition *= (self.args.transition_counter[tuple(self.args.corpus_tags[shift + idx - self.args.window_size: shift + idx + 1])] / self.args.transition_counter[new_tag])
        return transition

    def calc_prob(self, idx, word, prev_tag):
        """
        cala probability for each possible tag given all sequence. doesnt change any of data
        :param idx:
        :param word:
        :param prev_tag:
        :return:
        """
        tag_probs = list()
        for i in self.args.emission_counter[word].keys():
            tag = str(i)
            self.update_emmission(word, prev_tag, tag)
            emmission = self.calc_emmission(tag, word)

            self.update_transition(prev_tag, tag, idx)
            transition = self.calc_transition(idx, tag)

            self.reverse_emmission(word, prev_tag, tag)
            self.reverse_transition(prev_tag, tag, idx)

            tag_probs.append(emmission * transition)
        return tag_probs




    def match_tag_to_learned_tag(self):
        """
        match learned tags to gold as best as he can (not finished)
        :return:
        """
        tag_to_learned = defaultdict(Counter)
        for idx in self.args.indexes_untagged_words:
            tag_to_learned[self.args.gold[idx]][self.args.corpus_tags[idx]] += 1
        return tag_to_learned