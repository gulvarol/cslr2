"""
Functions for augmenting text data.
Implemented:
    - dropping words
    - swap words
    - shuffle sentences
"""
from typing import List

import numpy as np


class DropWords(object):
    """
    Drop words from a sentence.

    Args:
        p_sentence (float): probability of augmenting a sentence
        p_word (float): probability of dropping a word
    """
    def __init__(
        self,
        p_sentence: float = 0.5,
        p_word: float = 0.3,
    ) -> None:
        self.p_sentence = p_sentence
        self.p_word = p_word

    def _drop_words(self, sentence: str) -> str:
        # split sentence into words
        words = np.array(sentence.split())
        if len(words) == 1:
            return sentence
        kept_words = -1
        while kept_words < 1:
            # assign a random probability to each word
            words_probs = np.random.rand(len(words))
            # get indices of words to keep
            kept_indices = np.where(words_probs >= self.p_word)[0]
            kept_words = len(kept_indices)
        # keep only the words with indices in kept_indices
        kept_words = words[kept_indices]
        return " ".join(kept_words)

    def __call__(self, sentence: str) -> str:
        if np.random.rand() < self.p_sentence:
            return self._drop_words(sentence)
        return sentence


class SwapWords(object):
    """
    Swap words in a sentence.

    Args:
        nb_swaps (int): number of swaps
    """
    def __init__(self, nb_swaps: int = 2):
        self.nb_swaps = nb_swaps

    def __call__(self, sentence: str) -> str:
        # split sentence into words
        words = sentence.split()
        if len(words) == 1:
            return sentence
        for _ in range(self.nb_swaps):
            words = self._swap_words(words)
        return " ".join(words)

    def _swap_words(self, words: List[str]) -> List[str]:
        random_idx_1 = np.random.randint(len(words))
        random_idx_2 = np.random.randint(len(words))
        # could do identity swap
        words[random_idx_1], words[random_idx_2] = words[random_idx_2], words[random_idx_1]
        return words


class ShuffleWords(object):
    """
    Shuffle words in a sentence.

    Args:
        p_shuffle (float): probability of shuffling a sentence
    """
    def __init__(self, p_shuffle: float = 0.5):
        self.p_shuffle = p_shuffle

    def _shuffle_words(self, sentence: str) -> str:
        # split sentence into words
        words = sentence.split()
        if len(words) == 1:
            return sentence
        np.random.shuffle(words)
        return " ".join(words)

    def __call__(self, sentence: str) -> str:
        if np.random.rand() < self.p_shuffle:
            return self._shuffle_words(sentence)
        return sentence
