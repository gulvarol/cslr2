"""Python file defining functions to extract root words from a given word."""
import re
from typing import List, Union

import numpy as np
from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()


def get_root_words(
    words: Union[str, List[str], np.ndarray],
    segment_behavior: bool = False,
) -> List[List[str]]:
    """Normalise + lemmatise words"""
    if isinstance(words, List) or isinstance(words, np.ndarray):
        words = " ".join(words)
    if segment_behavior:
        words = words.strip().replace("-", " ").split()
        words = [word.lower() for word in words]
        words = [
            wordnet_lemmatizer.lemmatize(
                wordnet_lemmatizer.lemmatize(
                    " ".join(word.replace("/", " ").split()),
                    pos="v",
                ),
                pos="n",
            ) for word in words
        ]
    else:
        words = words.strip().split()
        words = [word.lower() for word in words]
        for word_idx, word in enumerate(words):
            true_words = re.split("-|/| ", word)
            words[word_idx] = [
                wordnet_lemmatizer.lemmatize(
                    wordnet_lemmatizer.lemmatize(true_word, pos="v"),
                    pos="n"
                ) for true_word in true_words if len(true_word) > 0
            ]
    return words
