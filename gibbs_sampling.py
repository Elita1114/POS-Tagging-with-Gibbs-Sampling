"""This file contains our implementation of the gibbs sampling algorithm."""

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Optional, Union
from mytypes import Emissions, Transitions, Tag, AbstractTag
from numpy.random import choice
from collections import defaultdict, Counter
from math_utils import normalize
from tqdm.auto import tqdm

from copy import deepcopy

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

    def run(
            self,
            epochs_number: int = 1000,
            save_at_end: bool = True,
            epochs_to_save_after: int = None,
            verbose: bool = True
    ) -> Tuple[Iterable[Tag], List[float]]:
        """Run the gibbs sampling algorithm"""

        assert epochs_number > 0, 'epochs_number has to be greater than 0.'

        scores: list = []

        _, score = self._match_tag_to_learned_tag()

        if verbose:
            print(f"\nEpoch: 0 (Before the sampling == Random init) | score: {score}")
        scores.append(score)

        for epoch in tqdm(range(1, epochs_number + 1)):
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

            if epochs_to_save_after is not None and epoch % epochs_to_save_after == 0:
                _, score = self._match_tag_to_learned_tag()
                if verbose:
                    print(f"\nEpoch: {epoch} | score: {score}")
                scores.append(score)

        mapping, score = self._match_tag_to_learned_tag()

        if verbose:
            print(f"\n-------------------------\n\nFinal score: {score}")

        # replace the symbols with the actual tags
        final_tags = self._swap_abstract_tags_and_true_tags(mapping)

        if save_at_end:
            save_object(self.args, 'saved_gibbs_args')

        return final_tags, scores


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

    def _match_tag_to_learned_tag(self, use_greedy_map: bool = True) -> Tuple[Dict[AbstractTag, Tag], float]:
        """
        Match the learned tags to the abstract tags we gave.

        :return: A mapping between the abstract tags to the tags we learned.
                 Also returns the score of the best mapping.
        """
        print("enter map")
        tags_we_learn_to_abstract_tags = defaultdict(NonNegativeCounter)

        for idx in self.args.indexes_of_untagged_words:
            tags_we_learn_to_abstract_tags[self.args.gold_tags[idx]][self.args.corpus_tags[idx]] += 1

        if use_greedy_map:
            return self._greedy_map(tags_we_learn_to_abstract_tags)

        return self._best_map(tags_we_learn_to_abstract_tags)

    def _best_map(self, tags_we_learn_to_abstract_tags) -> Tuple[Dict[AbstractTag, Tag], float]:
        denom: int = len(self.args.indexes_of_untagged_words)
        best_permutation = None
        best_score = -1

        for perm in tqdm(permutations(self.args.abstract_tags)):
            sum_ = 0
            for tag_we_learn, abstract_tag in zip(self.args.learning_tags, perm):
                sum_ += tags_we_learn_to_abstract_tags[tag_we_learn][str(abstract_tag)]

            score = sum_ / denom
            if best_score < score:
                best_score = score
                best_permutation = tuple(perm)

        mapping = {str(p): t for p, t in zip(best_permutation, self.args.learning_tags)}
        print("exit map")
        return mapping, best_score

    def _greedy_map(self, tags_we_learn_to_abstract_tags) -> Tuple[Dict[AbstractTag, Tag], float]:
        tags_we_learn_to_abstract_tags_copy = deepcopy(tags_we_learn_to_abstract_tags)
        mapping = dict()
        best_mapping = {'tag': None, 'abstract_tag': None, 'count': -1}
        score = 0.

        for i in range(len(tags_we_learn_to_abstract_tags.keys())):
            for actual_tag, abstract_tags_count in tags_we_learn_to_abstract_tags_copy.items():
                for abstract_tag, count in abstract_tags_count.items():
                    if count > best_mapping['count']:
                        best_mapping['tag'] = actual_tag
                        best_mapping['abstract_tag'] = abstract_tag
                        best_mapping['count'] = count

            mapping[best_mapping['abstract_tag']] = best_mapping['tag']
            score += best_mapping['count']
            tags_we_learn_to_abstract_tags_copy.pop(best_mapping['tag'], None)

            for abstract_tags_count in tags_we_learn_to_abstract_tags_copy.values():
                abstract_tags_count.pop(best_mapping['abstract_tag'], None)

            best_mapping['tag'] = None
            best_mapping['abstract_tag'] = None
            best_mapping['count'] = -1

        score /= len(self.args.indexes_of_untagged_words)

        return mapping, score

    def _swap_abstract_tags_and_true_tags(self, mapping: dict) -> List[Tag]:
        corpus_tags_copy = self.args.corpus_tags.copy()

        for idx in self.args.indexes_of_untagged_words:
            corpus_tags_copy[idx] = mapping[self.args.corpus_tags[idx]]

        return corpus_tags_copy
