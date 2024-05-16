"""Python file with all the functions used to work with the frame level evaluation dictionary."""
import os
import pickle
from copy import deepcopy
from operator import itemgetter
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from einops import rearrange
from tqdm import tqdm

from utils.cslr_metrics import get_labels_start_end_time
from utils.root_words import get_root_words
from utils.synonyms import synonym_combine

def pred_pickles_to_frame_level_predictions(
    pred_pickles: Union[List[str], str],
    id2word_dict: dict,
    synonyms: Optional[dict] = None,
    automatic_annotations: bool = False,
    remove_synonym_grouping: bool = False,
) -> dict:
    """
    Load predictions saved in pickle format.
    Convert to frame-level predictions.

    Args:
        pred_pickles (Union[List[str], str]): list of paths to pickle files
            or path to a single pickle file.
        id2word_dict (dict): dictionary mapping from index to word.
        synonyms (Optional[dict], optional): synonym dictionary. Defaults to None.
            If a synonym dictionary is provided, predictions will be combined between
            synonyms.
        automatic_annotations (bool): whether the predictions are automatic annotations.
            Defaults to False.
        remove_synonym_grouping (bool): whether to remove synonym grouping.

    Returns:
        dict: frame-level predictions with the following keys
            episode_name (List[str]): list of episode names
            sub_start (List[float]): list of start times
            sub_end (List[float]): list of end times
            labels (List[np.ndarray]): list of labels (frame-level)
            words (List[np.ndarray]): list of words (frame-level)
            probs (List[np.ndarray]): list of probabilities (frame-level)
            unique_key (List[str]): list of unique keys in format episode_name--start--end
    """
    out_dict = {
        "episode_name": [],
        "sub_start": [],
        "sub_end": [],
        "labels": [],
        "words": [],
        "probs": [],
        "unique_key": [],
    }
    if synonyms is not None:
        word2id_dict = {v: k for k, v in id2word_dict.items()}
    if isinstance(pred_pickles, str):
        pred_pickles = [pred_pickles]
    for pred_pickle in tqdm(pred_pickles):
        # episode name
        episode_name = os.path.basename(pred_pickle).replace(".pkl", "")
        # load predictions
        predictions = pickle.load(open(pred_pickle, "rb"))
        for timings, preds in predictions.items():
            # episode name
            out_dict["episode_name"].append(episode_name)
            # timings
            try:
                sub_start, sub_end = timings.split("--")
            except ValueError:
                sub_start, sub_end = timings.split("-")
            out_dict["sub_start"].append(float(sub_start))
            out_dict["sub_end"].append(float(sub_end))
            unique_key = f"{episode_name}--{float(sub_start):.2f}--{float(sub_end):.2f}"
            out_dict["unique_key"].append(unique_key)

            if not automatic_annotations:
                # labels and probds
                labels = np.array(preds["labels"][0])
                probs = np.array(preds["probs"][0])
                # check if batch size is not one
                if len(labels.shape) == 1:
                    labels = np.expand_dims(labels, axis=0)
                    probs = np.expand_dims(probs, axis=0)
                labels = rearrange(labels, "t k -> (t k)")
                words = itemgetter(*labels)(id2word_dict)
                words = rearrange(np.array(words), "(t k) -> t k", k=5)
                labels = rearrange(labels, "(t k) -> t k", k=5)
                if synonyms is not None and not remove_synonym_grouping:
                    # synonym grouping
                    new_words, new_probs = [], []
                    for word_top5, probs_top5 in zip(words, probs):
                        new_probs_top5, new_word_top5 = synonym_combine(
                            word_top5, probs_top5, synonyms
                        )
                        new_words.append(new_word_top5)
                        new_probs.append(new_probs_top5)
                    new_words = np.array(new_words)
                    probs = np.array(new_probs)
                    new_words = rearrange(new_words, "t k -> (t k)")
                    labels = itemgetter(*new_words)(word2id_dict)
                    labels = rearrange(np.array(labels), "(t k) -> t k", k=5)
                    words = rearrange(new_words, "(t k) -> t k", k=5)
                # only keep top 1
                labels, probs, words = labels[:, 0], probs[:, 0], words[:, 0]
            else:
                # labels and probs
                labels = np.array(preds["labels"])
                probs = np.array(preds["probs"])
                try:
                    if len(labels) == 1:
                        words = np.array([itemgetter(*labels)(id2word_dict)])
                    else:
                        words = np.array(itemgetter(*labels)(id2word_dict))
                except TypeError:
                    words = []
            # lemmatise words
            try:
                for word_idx, word in enumerate(words):
                    if " " in word:
                        words[word_idx] = word.replace(" ", "-")
                words = get_root_words(words)
                if len(words) != len(labels):
                    raise ValueError("Length mismatch")
                assert len(words) == len(labels), print(words)
            except TypeError:
                pass
            out_dict["labels"].append(labels)
            out_dict["probs"].append(probs)
            out_dict["words"].append(words)
    return out_dict


def gt_csvs_to_frame_level_gt(
    gt_csvs: Union[List[str], str],
    fps: int = 25,
) -> dict:
    """
    Load ground truth saved in csv format.
    Convert to frame-level predictions.

    Args:
        gt_csvs (Union[List[str], str]): list of paths to csv files.

    Returns:
        dict: frame-level ground truth with the following keys
            episode_name (List[str]): list of episode names
            sub_start (List[float]): list of start times
            sub_end (List[float]): list of end times
            frame_ground_truth (List[np.ndarray]): list of ground truth (frame-level)
            segment_ground_truth (List[str]): list of ground truth (segment-level)
            raw_segment_ground_truth (List[List[str]]): list of ground truth
                                                    (segment level, without collapsing)
            subtitles (List[np.ndarray]): list of subtitles
            unique_key (List[str]): list of unique keys in format episode_name--start--end
    """
    out_dict = {
        "episode_name": [],
        "sub_start": [],
        "sub_end": [],
        "frame_ground_truth": [],
        "segment_ground_truth": [],
        "raw_segment_ground_truth": [],
        "subtitles": [],
        "unique_key": [],
    }

    if isinstance(gt_csvs, str):
        gt_csvs = [gt_csvs]
    for gt_csv in tqdm(gt_csvs):
        # episode_name
        episode_name = os.path.basename(gt_csv).replace(".csv", "")
        # load
        try:
            gt_df = pd.read_csv(gt_csv, delimiter=",")
            starts, ends = gt_df["start_sub"].tolist(
            ), gt_df["end_sub"].tolist()
            subs = gt_df["english sentence"].tolist()
            gt_glosses = gt_df["approx gloss sequence"].tolist()
            assert len(starts) == len(ends) and len(ends) == len(
                subs) and len(subs) == len(gt_glosses)
            for start, end, sub, gt_gloss in zip(starts, ends, subs, gt_glosses):
                if isinstance(gt_gloss, float) or start >= end:
                    # no gt
                    # the second condition should not happen with the new fix_alignement.py
                    # temporary fix
                    pass
                else:
                    # episode name
                    out_dict["episode_name"].append(episode_name)
                    out_dict["sub_start"].append(float(start))
                    out_dict["sub_end"].append(float(end))
                    unique_key = f"{episode_name}--{start:.2f}--{end:.2f}"
                    out_dict["unique_key"].append(unique_key)
                    # frame-level ground truth
                    gt_labels, gt_segment, gt_segment_raw = gloss_update(
                        gt_gloss,
                        start,
                        end,
                        fps=fps,
                    )
                    out_dict["frame_ground_truth"].append(gt_labels)
                    out_dict["segment_ground_truth"].append(gt_segment)
                    out_dict["raw_segment_ground_truth"].append(gt_segment_raw)
                    out_dict["subtitles"].append(sub)
        except Exception as gt_exception:
            print(f"Error with {gt_csv}: {gt_exception}")
    return out_dict


def populate_combined_dict(
    combined_dictionary: dict,
    input_dictionary: dict,
    setname: str,
) -> dict:
    """
    Populate combined dictionary with input dictionary.

    Args:
        combined_dictionary (dict): combined dictionary
        input_dictionary (dict): input dictionary
        setname (str): set name, either "gt" or "pred"

    Returns:
        combined_dictionary: populated combined dictionary
    """
    for key, value in input_dictionary.items():
        if key in ["episode_name", "sub_start", "sub_end", "unique_key"]:
            key += f"_{setname}"
        combined_dictionary[key] = value
    return combined_dictionary


def combine_gt_pred_dict(
    gt_dictionary: dict,
    pred_dictionary: dict,
) -> dict:
    """
    Combine ground truth and predictions into a single dictionary.

    Args:
        gt_dictionary (dict): ground truth dictionary
        pred_dictionary (dict): predictions dictionary

    Returns:
        combined_dictionary: combined dictionary
    """
    combined_dictionary = {}
    copy_pred_dictionary = deepcopy(pred_dictionary)

    pred_unique_keys = np.array(copy_pred_dictionary["unique_key"])
    gt_unique_keys = np.array(gt_dictionary["unique_key"])
    filtering, mapping = np.where(
        pred_unique_keys[:, None] == gt_unique_keys[None, :]
    )

    # first filter out the predictions that don't have ground truth
    for key, value in copy_pred_dictionary.items():
        copy_pred_dictionary[key] = np.array(
            value, dtype=object,
        )[filtering].tolist()
    # next, order gt to match pred in terms of order
    for key, value in gt_dictionary.items():
        gt_dictionary[key] = np.array(value, dtype=object)[
            mapping].tolist()
    # then populate the combined dictionary
    combined_dictionary = populate_combined_dict(
        combined_dictionary, gt_dictionary, "gt",
    )
    combined_dictionary = populate_combined_dict(
        combined_dictionary, copy_pred_dictionary, "pred",
    )
    return combined_dictionary


def save_all_annots(
    combined_dictionary: dict,
) -> dict:
    """
    Goes through all annotations and saves them in a dictionary.

    Args:
        combined_dictionary (dict): combined dictionary

    Returns:
        all_annots (dict): dictionary of all annotations. Keys are words, values are counts.
    """
    all_annots = {}
    frame_wise_gt_labels = combined_dictionary["frame_ground_truth"]
    for gt_labels in frame_wise_gt_labels:
        gt_segments, _, _ = get_labels_start_end_time(
            gt_labels,
            bg_class=["no annotation"],
        )
        for segment in gt_segments:
            for word in segment:
                try:
                    all_annots[word] += 1
                except KeyError:
                    all_annots[word] = 1
    return all_annots

def gloss_update(
    gloss: str,
    start: Union[str, float],
    end: Union[str, float],
    fps: int,
    stars: bool = False,
):
    """
    Computes frame-level ground truth from gloss (with timings).

    Args:
        gloss (str): string with glosses along timings
        start (Union[str, float]): start time
        end (Union[str, float]): end time
        fps (int): fps of the video
        stars (bool): whether behaviour has star annotations loading.
            Defaults to False.

    Returns:
        labels (List[List[str]]): frame-level ground truth
        segment (str): segment-level ground truth in one string
        segment_raw (List[List[str]]): segment-level ground truth without collapsing
    """
    labels = [["no annotation"]] * \
        int((float(end) - float(start)) * fps)
    timings = gloss.replace("]", "[").replace("'", "")
    raw_annots = timings.split("[")[:-1][::2]
    segment_raw = []
    for raw_annot in raw_annots:
        if (stars and "*" in raw_annot) or not stars:
            raw_annot = raw_annot.split("/")
            segment = []
            for annot in raw_annot:
                if len(annot) > 0:
                    if annot[0] == " ":
                        annot = annot[1:]
                    segment.append(annot)
            segment_raw.append(segment)
    timings = timings.replace(" ", "/").replace("--", "-")
    timings = timings.split('[')[:-1]
    annots = timings[::2]
    # lemmatise annotations
    annots = [
        annot if annot[0] != "/" else annot[1:]
        for annot in annots
    ]
    annots_seg = get_root_words(annots, True)
    annots = get_root_words(annots)
    times = timings[1::2]
    assert len(annots) == len(times)
    for annot, time in zip(annots, times):
        has_star = True
        if stars:
            has_star = any(["*" in ann for ann in annot])
        if has_star:
            start_time, end_time = time.split("-")
            start_time = float(start_time)
            end_time = float(end_time)
            start_idx = int(
                (start_time - float(start)) * fps
            )
            end_idx = min(
                len(labels),
                int((end_time - float(start)) * fps)
            )
            annot_len = end_idx - start_idx
            if annot_len < 1:
                # no annotation
                pass
            else:
                labels[start_idx:end_idx] = [annot] * annot_len
    return labels, " ".join(annots_seg), segment_raw
