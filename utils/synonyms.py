"""Functions related to synonyms."""
from typing import List, Union

import numpy as np
import torch


def synonym_combine(
    labels: Union[torch.tensor, np.ndarray],
    probs: Union[torch.tensor, np.ndarray],
    synonyms: dict,
    verbose: bool = False,
):
    """
    Function aggregating probabilities of synonyms.

    Args:
        labels (Union[torch.tensor, np.ndarray]): list of labels.
        probs (Union[torch.tensor, np.ndarray]): list of probabilities.
        synonyms (dict): dictionary of synonyms.
        verbose (bool): whether to print the output.

    Returns:
        Union[torch.tensor, np.ndarray]: aggregated probabilities.
        Union[torch.tensor, np.ndarray]: aggregated labels.
    """
    change = False
    new_probs = []
    for anchor_idx, anchor in enumerate(labels):
        try:
            anchor = anchor.replace("-", " ")
            syns = synonyms[anchor]
            if verbose:
                print(anchor, syns)
            anchor_new_prob = 0
            for checked_idx, checked_label in enumerate(labels):
                checked_label = checked_label.replace("-", " ")
                if checked_label in syns:
                    if verbose:
                        print(checked_label)
                    anchor_new_prob += probs[checked_idx]
                if checked_idx != anchor_idx:
                    change = True
            new_probs.append(anchor_new_prob)
        except KeyError:
            # prediction not in the synonym list
            new_probs.append(probs[anchor_idx])
    if change:
        # need to sort
        sorted_indices = np.argsort(- 1 * np.array(new_probs))
        if verbose:
            print(labels, new_probs)
        new_probs = np.array(new_probs)[sorted_indices]
        labels = np.array(labels)[sorted_indices]
        if verbose:
            print(labels, new_probs)
    else:
        new_probs = np.array(new_probs)
        labels = np.array(labels)
    return new_probs, labels


def fix_synonyms_dict(synonyms: dict, verbose: bool = False) -> dict:
    """
    Make sure that the synonyms dictionary satisfies the following:
        - if a is a synonym of b, then b is a synonym of a
        - a is a synonym of a

    Args:
        synonyms (dict): dictionary of synonyms.
        verbose (bool): whether to print the output.

    Returns:
        dict: updated dictionary of synonyms.
    """
    change_count = 0
    syn_change_count = 0
    for word, syns in synonyms.items():
        if word not in syns:
            change_count += 1
            syns.append(word)
            # need to check that for each synonym in the list, the word is in the list of synonyms for that synonym
        for syn in syns:
            if word not in synonyms[syn]:
                synonyms[syn].append(word)
                syn_change_count += 1
    if verbose:
        print(f"Added {change_count} words to their own synonyms list")
        print(f"Total number of words: {len(synonyms)}")
        print(f"Added {syn_change_count} so that a is a syn of b is equivalent to b is a syn of a")
    return synonyms


def extend(labels: List[str], synonyms: dict) -> List[str]:
    """
    Extend a list of labels using synonyms

    Args:
        labels (List[str]): list of labels
        synonyms (dict): synonym dictionary

    Returns:
        new_labels (List[str]): list of labels with synonyms
    """
    new_labels = []
    for lbl in labels:
        if synonyms is not None and lbl in synonyms.keys():
            temp_synonyms = synonyms[lbl]
            if lbl not in temp_synonyms:
                temp_synonyms.append(lbl)
            new_labels.extend(temp_synonyms)
        else:
            # no synonyms or no synonyms for this word
            new_labels.extend([lbl])
    return new_labels
