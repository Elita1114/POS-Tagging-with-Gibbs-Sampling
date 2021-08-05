"""This file contains our implementation of the gibbs sampling algorithm."""

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Optional
from mytypes import Emissions, Transitions, Tag
from numpy.random import choice
from collections import defaultdict, Counter
from math_utils import normalize
from tqdm import tqdm

from itertools import permutations

from non_negative_counter import NonNegativeCounter

from general_utils import save_object, load_object


@dataclass(eq=False)
class GibbsSamplingArguments(object):
    """
    This class is used to store the gibbs sampling arguments.
    To use the *GibbsSampler* class you have to create an instance
    of this class and pass to the *GibbsSampler* class.
    """
    corpus_words: List[str]
    corpus_tags: List[Tag]
    indexes_of_untagged_words: List[int]
    emission_probs: Emissions
    emission_counter: Emissions
    transition_counter: Transitions
    learning_tags: Iterable[Tag]
    abstract_tags: Iterable[Tag]
    window_size: int
    gold_tags: List[Tag]


class GibbsSampler(object):
    def __init__(self, args: GibbsSamplingArguments):
        assert args is not None, "args can't be None."
        self.args = args

    def run(self, epochs_number: int = 1000, iter_to_log_score: Optional[int] = 10, save_at_end: bool = True) -> Iterable[Tag]:
        """Run the gibbs sampling algorithm"""

        assert epochs_number > 0, 'epochs_number has to be greater than 0.'

        if iter_to_log_score is not None:
            assert iter_to_log_score > 0, 'iter_to_log_score has to be greater than 0.'

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

                # update the emission and transition data structures
                self._update_emission(word, prev_tag, new_tag, change_emission_probs=True)
                self._update_transition(prev_tag, new_tag, idx)

                # set the new tag
                self.args.corpus_tags[idx] = new_tag

                if iter_to_log_score is not None and idx % iter_to_log_score == 0:
                    _, score = self._match_tag_to_learned_tag()
                    print(f"score: {score}")

        mapping, score = self._match_tag_to_learned_tag()

        # replace the symbols with the actual tags
        final_tags = self._swap_abstract_tags_and_true_tags(mapping)

        if save_at_end:
            save_object(self.args, 'saved_gibbs_args')

        return final_tags

    @staticmethod
    def load_from_checkpoint(checkpoint: str):
        args = load_object(checkpoint)

        if type(args) != GibbsSamplingArguments:
            raise RuntimeError("The given checkpoint file does not contain valid arguments.")

        return GibbsSampler(args=args)

    @staticmethod
    def _draw_tag(tag_probs) -> Tag:
        """Draw a tag from a probability vector."""
        return str(choice(range(len(tag_probs)), 1, p=tag_probs)[0])

    def _calc_emission(self, word: str, tag: Tag) -> float:
        """
        Calculate the emission probability from the given word to the given tag.
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
        Change the counts of the emission counter with the new tag.
        """
        self.args.emission_counter[word][prev_tag] -= 1
        self.args.emission_counter[word][new_tag] += 1

        if change_emission_probs:
            self.args.emission_probs[word] = normalize(self.args.emission_counter[word])

    def _reverse_emission(self, word, prev_tag, new_tag) -> None:
        """
        When called after *update_emission* (with change_emission_probs = False),
        it cancels all of its changes.
        """
        self.args.emission_counter[word][prev_tag] += 1
        self.args.emission_counter[word][new_tag] -= 1

    def _update_transition(self, prev_tag: Tag, new_tag: Tag, idx: int) -> None:
        """
        Change the counts of the transition with the new tag.
        """
        self.args.transition_counter[prev_tag] -= 1

        for shift in range(0, self.args.window_size + 1):
            self.args.transition_counter[
                tuple(
                    self.args.corpus_tags[shift + idx - self.args.window_size: shift + idx + 1]
                )
            ] -= 1

        self.args.transition_counter[new_tag] += 1
        self.args.corpus_tags[idx] = new_tag

        for shift in range(0, self.args.window_size + 1):
            self.args.transition_counter[
                tuple(self.args.corpus_tags[shift + idx - self.args.window_size: shift + idx + 1])
            ] += 1

        self.args.corpus_tags[idx] = prev_tag

    def _reverse_transition(self, prev_tag: Tag, new_tag: Tag, idx: int) -> None:
        """
        When called after *update_transition*, it cancels all of its changes.
        """
        self.args.transition_counter[prev_tag] += 1

        for shift in range(0, self.args.window_size + 1):
            self.args.transition_counter[
                tuple(self.args.corpus_tags[shift + idx - self.args.window_size: shift + idx + 1])
            ] += 1

        self.args.transition_counter[new_tag] -= 1
        self.args.corpus_tags[idx] = new_tag

        for shift in range(0, self.args.window_size + 1):
            self.args.transition_counter[
                tuple(self.args.corpus_tags[shift + idx - self.args.window_size: shift + idx + 1])
            ] -= 1

        self.args.corpus_tags[idx] = prev_tag

    def _calc_transition_prob(self, idx: int, prev_tag: Tag, new_tag: Tag) -> float:
        """
        Calculate the transition probability.
        """
        transition: float = 1.0

        self.args.corpus_tags[idx] = new_tag

        for shift in range(self.args.window_size + 1):
            transition *= (
                    self.args.transition_counter[tuple(
                               self.args.corpus_tags[shift + idx - self.args.window_size: shift + idx + 1]
                    )] / self.args.transition_counter[new_tag]
            )

        self.args.corpus_tags[idx] = prev_tag

        return transition

    def _calc_tag_probs(self, word_and_tag_idx: int, word: str, prev_tag: Tag) -> List[float]:
        """
        Calculate the probability for each possible tag given all sequence.
        This function doesn't change any of the data.
        """
        tag_probs = list()

        for tag in self.args.abstract_tags:
            tag = str(tag)
            self._update_emission(word, prev_tag, tag)
            emission = self._calc_emission(word, tag)

            self._update_transition(prev_tag, tag, word_and_tag_idx)
            transition = self._calc_transition_prob(word_and_tag_idx, prev_tag, tag)

            self._reverse_emission(word, prev_tag, tag)
            self._reverse_transition(prev_tag, tag, word_and_tag_idx)

            tag_probs.append(emission * transition)
        return normalize(tag_probs)

    def _match_tag_to_learned_tag(self) -> Tuple[Dict[Tag, Tag], float]:
        """
        Match the learned tags to the abstract tags we gave.

        :return: A mapping between the abstract tags to the tags we learned.
                 Also returns the score of the best mapping.
        """
        tags_we_learn_to_abstract_tags = defaultdict(NonNegativeCounter)
        for idx in self.args.indexes_of_untagged_words:
            tags_we_learn_to_abstract_tags[self.args.gold_tags[idx]][self.args.corpus_tags[idx]] += 1

        options_to_score = dict()

        denom: int = len(self.args.indexes_of_untagged_words)

        for perm in permutations(self.args.abstract_tags):
            sum_ = 0
            for tag_we_learn, abstract_tag in zip(self.args.learning_tags, perm):
                sum_ += tags_we_learn_to_abstract_tags[tag_we_learn][abstract_tag]

            score = sum_ / denom
            options_to_score[tuple(perm)] = score

        best_permutation = max(options_to_score.keys(), key=lambda key: options_to_score[key])
        mapping = {p: t for p, t in zip(best_permutation, self.args.learning_tags)}

        return mapping, options_to_score[best_permutation]

    def _swap_abstract_tags_and_true_tags(self, mapping: dict) -> List[Tag]:
        corpus_tags_copy = self.args.corpus_tags.copy()

        for idx in self.args.indexes_of_untagged_words:
            corpus_tags_copy[idx] = mapping[self.args.corpus_tags[idx]]

        return corpus_tags_copy
