"""This file contains our implementation of the gibbs sampling algorithm."""

from dataclasses import dataclass
from typing import Union, Iterable
from mytypes import Emissions, Transitions, Tag
from numpy.random import choice
from collections import defaultdict, Counter
from math_utils import normalize
from tqdm import tqdm


@dataclass(eq=False)
class GibbsSamplingArguments(object):
    """
    This class is used to store the gibbs sampling arguments.
    To use the *GibbsSampler* class you have to create an instance
    of this class and pass to the *GibbsSampler* class.
    """

    corpus_words: Iterable
    corpus_tags: Iterable
    indexes_of_untagged_words: Iterable
    emission_probs: Emissions
    emission_counter: Emissions
    transition_counter: Transitions
    learning_tags: Iterable
    window_size: int
    gold_tags: Iterable


class GibbsSampler(object):
    def __init__(self, args: GibbsSamplingArguments):
        assert args is not None, "args can't be None."
        self.args = args

    def run(self, epochs_number: int = 1000) -> Iterable[Tag]:
        """Run the gibbs sampling algorithm"""

        assert epochs_number > 0, 'epochs_number has to be greater than 0.'

        for _ in tqdm(range(epochs_number)):
            for idx in self.args.indexes_of_untagged_words:
                # get the word
                word = self.args.corpus_words[idx]

                # keep the previous tag of this word
                prev_tag = self.args.corpus_tags[idx]

                # calculate the tag probabilities
                tag_probs = self._calc_tag_probs(idx, word, prev_tag)

                # draw a new tag
                new_tag = self._draw_tag(tag_probs)

                # set the new tag
                self.args.corpus_tags[idx] = new_tag

                # update the emission and transition data structures
                self._update_emission(word, prev_tag, new_tag, change_emission_probs=True)
                self._update_transition(prev_tag, new_tag, idx)

        tag_to_learned = self.match_tag_to_learned_tag()

        # replace the symbols with the actual tags

        return self.args.corpus_tags

    @staticmethod
    def _draw_tag(tag_probs) -> str:
        """Draw a tag from a probability vector."""
        return str(choice(range(len(tag_probs)), 1, p=tag_probs)[0])

    def _calc_emission(self, tag: Tag, word: str) -> float:
        """
        Calculate the emission probability from the given tag to the given word.
        """
        emission_probs_local = normalize(self.args.emission_counter[word])
        return emission_probs_local[tag]

    def _update_emission(
            self,
            word: str,
            prev_tag: Tag,
            new_tag: Tag,
            change_emission_probs: bool = False
    ) -> None:
        """
        Change the counts of emission with the new tag.
        """

        if self.args.emission_counter[word][prev_tag] > 0:
            self.args.emission_counter[word][prev_tag] -= 1
        else:
            raise RuntimeError(f"Emissions counter has a negative entry for word '{word}'"
                               f" and tag '{prev_tag}'.")

        self.args.emission_counter[word][new_tag] += 1

        if change_emission_probs:
            self.args.emission_probs[word] = normalize(self.args.emission_counter[word])

    def _reverse_emission(self, word, prev_tag, new_tag):
        """
        When called after *update_emission* (with change_emission_probs = False),
        it will cancel all of its changes.
        """
        self.args.emission_counter[word][prev_tag] += 1

        if self.args.emission_counter[word][prev_tag] > 0:
            self.args.emission_counter[word][new_tag] -= 1
        else:
            raise RuntimeError(f"Emissions counter has a negative entry for word '{word}'"
                               f" and tag '{new_tag}'.")

    def _update_transition(self, prev_tag: Tag, new_tag: Tag, idx: int) -> None:
        """
        Change the counts of the transition with the new tag.
        """
        if self.args.transition_counter[prev_tag] > 0:
            self.args.transition_counter[prev_tag] -= 1
        else:
            raise RuntimeError(f"Transitions counter has a negative entry for tag '{prev_tag}'.")

        for shift in range(self.args.window_size + 2):
            self.args.transition_counter[
                tuple(self.args.corpus_tags[shift + idx - self.args.window_size: shift + idx + 1])
            ] -= 1

        self.args.transition_counter[new_tag] += 1
        self.args.corpus_tags[idx] = new_tag

        for shift in range(self.args.window_size + 2):
            self.args.transition_counter[
                tuple(self.args.corpus_tags[shift + idx - self.args.window_size: shift + idx + 1])
            ] += 1

        self.args.corpus_tags[idx] = prev_tag

    def reverse_transition(self, prev_tag, new_tag, idx):
        """
        When called after *update_transition*, it will cancel all of its changes.
        """
        self.args.transition_counter[prev_tag] += 1

        for shift in range(self.args.window_size + 2):
            self.args.transition_counter[
                tuple(self.args.corpus_tags[shift + idx - self.args.window_size: shift + idx + 1])
            ] += 1

        self.args.transition_counter[new_tag] -= 1
        self.args.corpus_tags[idx] = new_tag

        for shift in range(self.args.window_size + 2):
            self.args.transition_counter[
                tuple(self.args.corpus_tags[shift + idx - self.args.window_size: shift + idx + 1])
            ] -= 1

        self.args.corpus_tags[idx] = prev_tag

    def _calc_transition_prob(self, idx, new_tag) -> float:
        """
        Calculate the transition probability.
        """
        transition: float = 1.0

        for shift in range(self.args.window_size + 2):
            transition *= (self.args.transition_counter[
                               tuple(self.args.corpus_tags[shift + idx - self.args.window_size: shift + idx + 1])
                           ] / self.args.transition_counter[new_tag])

        return transition

    def _calc_tag_probs(self, idx, word, prev_tag):
        """
        Calculate the probability for each possible tag given all sequence.
        This function doesn't change any of the data.
        """
        tag_probs = list()

        for i in self.args.emission_counter[word].keys():
            tag = str(i)
            self._update_emission(word, prev_tag, tag)
            emission = self._calc_emission(tag, word)

            self._update_transition(prev_tag, tag, idx)
            transition = self._calc_transition_prob(idx, tag)

            self._reverse_emission(word, prev_tag, tag)
            self.reverse_transition(prev_tag, tag, idx)

            tag_probs.append(emission * transition)
        return normalize(tag_probs)

    def match_tag_to_learned_tag(self):
        """
        match learned tags to gold as best as he can (not finished)
        :return:
        """
        # TODO finish this method
        tag_to_learned = defaultdict(Counter)
        for idx in self.args.indexes_of_untagged_words:
            tag_to_learned[self.args.gold_tags[idx]][self.args.corpus_tags[idx]] += 1
        return tag_to_learned
