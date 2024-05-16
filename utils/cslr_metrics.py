"""
Python file defining metrics to evaluate for cslr.

Levenstein edit distance, precision, recall, and F1 score.
"""
from copy import deepcopy
from typing import List, Optional, Union

import numpy as np

from utils.synonyms import extend

def levenstein(
    predicted_segments: List[str],
    gt_segments: List[str],
    norm: bool = False,
    synonyms: Optional[dict] = None,
):
    """
    Levenstein edit distance.

    Args:
        predicted_segments (List[str]): predicted labels (segment-level)
        gt_segments (List[str]): ground truth labels (segment-level)
        norm (bool, optional): whether to normalise the score. Defaults to False.
        synonyms (Optional[dict], optional): synonym dictionary. Defaults to None.

    Returns:
        score: score
        n_col: length of the ground truth
    """
    m_row = len(predicted_segments)
    n_col = len(gt_segments)
    cost_matrix = np.zeros([m_row + 1, n_col + 1], float)

    # intial weights of the cost matrix
    cost_matrix[:, 0] = np.arange(m_row + 1)
    cost_matrix[0, :] = np.arange(n_col + 1)

    for j in range(1, n_col + 1):
        for i in range(1, m_row + 1):
            predicted_labels = predicted_segments[i - 1]
            gt_labels = gt_segments[j - 1]
            # extend with synonyms
            gt_labels = extend(labels=gt_labels, synonyms=synonyms)
            # equality checks
            equality_check = any(
                [predicted_lbl in gt_labels for predicted_lbl in predicted_labels])
            reverse_equality_check = False
            if equality_check or reverse_equality_check:
                cost_matrix[i, j] = cost_matrix[i - 1, j - 1]
            else:
                cost_matrix[i, j] = min(
                    cost_matrix[i - 1, j] + 1,
                    cost_matrix[i, j - 1] + 1,
                    cost_matrix[i - 1, j - 1] + 1
                )
    if norm:
        score = (1 - cost_matrix[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = cost_matrix[-1, -1]
    return score, n_col


def compute_frame_level_metrics(
    combined_dictionary: dict,
    synonyms: Optional[dict] = None,
    stride: int = 2,
    tau: float = 0.0,
    do_print: bool = False,
) -> List[float]:
    """
    Compute frame-level metrics.
    That is, compute frame-level accuracy, precision, recall, and F1.

    Args:
        combined_dictionary (dict): combined dictionary.
        synonyms (Optional[dict], optional): synonym dictionary. Defaults to None.
        stride (int): striding factor for predictions. Defaults to 2.
        tau (float): threshold for predictions. Defaults to 0.0.
        do_print (bool): whether to print the metrics. Defaults to False.

    Returns:
        List[float]: frame-level metrics
    """
    frame_accuracy = {"correct": [], "total": [], "total_predicted": []}
    predicted_words = combined_dictionary["words"]
    predicted_probs = combined_dictionary["probs"]
    ground_truth_labels = combined_dictionary["frame_ground_truth"]

    for pred, prob, gt_label in zip(predicted_words, predicted_probs, ground_truth_labels):
        correct = 0
        total_predicted = 0
        for pred_idx, pred_word in enumerate(pred):
            prob_word = prob[pred_idx]
            if prob_word >= tau:
                gt_indices = [pred_idx * stride + i for i in range(stride)]
                for gt_idx in gt_indices:
                    if gt_idx < len(gt_label) and gt_label[gt_idx] != "no annotation":
                        total_predicted += 1
                        for word in pred_word:
                            target_words = gt_label[gt_idx]
                            if isinstance(target_words, str):
                                target_words = [target_words]
                            if synonyms is not None:
                                # need to extend target words with synonyms
                                new_target_words = deepcopy(target_words)
                                for target_word in target_words:
                                    if target_word in synonyms.keys():
                                        new_target_words.extend(
                                            synonyms[target_word])
                                target_words = new_target_words
                            if word in target_words:
                                correct += 1
                                break
        total = len(gt_label) - gt_label.count("no annotation")
        frame_accuracy["correct"].append(correct)
        frame_accuracy["total"].append(total)
        frame_accuracy["total_predicted"].append(total_predicted)

    if do_print:
        print(f"Accuracies with threshold {tau}:")
        print(
            f"Total correct: {sum(frame_accuracy['correct'])}, Total: {sum(frame_accuracy['total'])}, percentage: {sum(frame_accuracy['correct']) / sum(frame_accuracy['total']) * 100:.2f}%")  # pylint: disable=line-too-long
        print(f"Total correct: {sum(frame_accuracy['correct'])}, Total: {sum(frame_accuracy['total_predicted'])}, percentage: {sum(frame_accuracy['correct']) / sum(frame_accuracy['total_predicted']) * 100:.2f}%")  # pylint: disable=line-too-long

    precision = sum(frame_accuracy["correct"]) / \
        sum(frame_accuracy["total_predicted"])
    recall = sum(frame_accuracy["correct"]) / sum(frame_accuracy["total"])
    return precision, recall


def edit_score(
    frame_wise_predictions: List[List[str]],
    frame_wise_gt_labels: List[List[str]],
    frame_wise_probs: Optional[List[float]] = None,
    norm: bool = True,
    bg_class: List[str] = ["no annotation"],
    synonyms: Optional[dict] = None,
    tau: float = 0.0,
    min_count: int = 1,
):
    """
    Compute edit score from combined dictionary.

    Args:
        frame_wise_predictions (List[List[str]]): list of frame-wise predictions.
        frame_wise_gt_labels (List[List[str]]): list of frame-wise ground truth labels.
        frame_wise_probs (Optional[List[float]], optional): list of frame-wise probabilities.
            Defaults to None.
        norm (bool): whether to normalise the score. Defaults to True.
        bg_class (List[str], optional): list of all classes in frame_wise_labels to ignore.
        synonyms (Optional[dict], optional): synonym dictionary. Defaults to None.
        tau (float): threshold for predictions. Defaults to 0.0.
        min_count (int): minimum count for segmenting. Defaults to 1.

    Returns:
        score: score
        n_col: length of the ground truth
    """
    gt_segments, _, _ = get_labels_start_end_time(
        frame_wise_gt_labels,
        bg_class=bg_class,
        synonyms=synonyms,
    )
    predicted_segments = pred_filter(
        frame_wise_predictions,
        frame_wise_probs=frame_wise_probs,
        bg_class=bg_class,
        synonyms=synonyms,
        tau=tau,
        min_count=min_count,
    )
    return levenstein(
        predicted_segments,
        gt_segments,
        norm,
        synonyms=synonyms,
    )


def get_labels_start_end_time(
    frame_wise_labels: Union[List[str], List[List[str]]],
    frame_wise_probs: Optional[List[float]] = None,
    bg_class: List[str] = ["no annotation"],
    synonyms: Optional[dict] = None,
    tau: float = 0.0,
):
    """
    Get list of start and end times of each interval / segment.

    Args:
        frame_wise_labels (Union[List[str], List[List[str]]]): list of frame-wise labels.
        frame_wise_probs (Optional[List[float]], optional): list of frame-wise probabilities.
            Defaults to None.
        bg_class (List[str], optional): list of all classes in frame_wise_labels to ignore.
        synonyms (Optional[dict], optional): synonym dictionary. Defaults to None.
        tau (float): threshold for predictions. Defaults to 0.0.

    Returns:
        labels: list of labels of the segments
        starts: list of start times of the segments
        ends: list of end times of the segments

    """
    labels, starts, ends = [], [], []
    last_label = frame_wise_labels[0]  # is a list
    last_labels = last_label

    last_labels = extend(labels=last_labels, synonyms=synonyms)

    last_check = [lbl not in bg_class for lbl in last_labels]
    if any(last_check):
        labels.append(last_label)
        starts.append(0)
    for label_idx, label in enumerate(frame_wise_labels):
        prob = frame_wise_probs[label_idx] if frame_wise_probs is not None else 1.0
        change = False
        if not change:
            if prob < tau:
                # remove prediction if probability is too low
                label = ["no annotation"]

        previous_check = [lbl in last_labels for lbl in label]
        # check that the current label is not part of the same segment
        # as the ongoing segment
        if not any(previous_check):
            # new segment
            # need to check if it is a lexical segment
            # or a non-lexical segment (i.e. in bg_class)
            new_check = [lbl not in bg_class for lbl in label]
            if any(new_check):
                # new lexical segment
                # need to append the new start
                labels.append(label)
                starts.append(label_idx)
            # need to check if the ongoing segment is a lexical segment
            # if so, need to close it
            # else, no need to close it (never opened)
            last_check = [lbl not in bg_class for lbl in last_labels]
            if any(last_check):
                ends.append(label_idx)
            last_label = label
            last_labels = last_label
            last_labels = extend(labels=last_labels, synonyms=synonyms)
    last_check = [lbl not in bg_class for lbl in last_labels]
    if any(last_check):
        ends.append(len(frame_wise_labels))

    if len(labels) != len(starts) or len(labels) != len(ends) or len(starts) != len(ends):
        if len(ends) < len(starts):
            print("Ends are less than starts")
            try:
                ends = [starts[1]] + ends
            except:
                raise ValueError("Ends are less than starts")
        elif len(starts) < len(ends):
            raise ValueError("Starts are less than ends")
    return labels, starts, ends

def pred_filter(
    frame_wise_predictions: List[List[str]],
    frame_wise_probs: Optional[List[float]] = None,
    bg_class: List[str] = ["no annotation"],
    synonyms: Optional[dict] = None,
    tau: float = 0.0,
    min_count: int = 1,
    merge_consecutive: bool = False,
    return_all: bool = False,
    stride: int = 2,
):
    """
    Create segments out of frame_wise_predictions,
    with filtering steps (thresholding, min count).

    Args:
        frame_wise_predictions (List[List[str]]): list of frame-wise predictions.
        frame_wise_probs (Optional[List[float]], optional): list of frame-wise probabilities.
            Defaults to None.
        bg_class (List[str], optional): list of all classes in frame_wise_labels to ignore.
        synonyms (Optional[dict], optional): synonym dictionary. Defaults to None.
        tau (float): threshold for predictions. Defaults to 0.0.
        min_count (int): minimum count for segmenting. Defaults to 1.
        merge_consecutive (bool): whether to merge consecutive segments with synonyms.
            Defaults to False.
        return_all (bool): whether to return segments with time stamps. Defaults to False.
        stride (int): striding factor for predictions. Defaults to 2.

    Returns:
        predicted_segments (List[List[str]]): list of predicted segments
        predicted_starts (List[float]): list of start times of the segments (if return_all)
        predicted_ends
    """
    try:
        predicted_segments, predicted_starts, predicted_ends = get_labels_start_end_time(
            frame_wise_predictions,
            frame_wise_probs=frame_wise_probs,
            bg_class=bg_class,
            synonyms=synonyms,
            tau=tau,
        )
    except IndexError:
        # no prediction
        predicted_segments, predicted_starts, predicted_ends = [], [], []
    if min_count > 1:
        # need to remove segments that are too short
        diff = np.array(predicted_ends) - np.array(predicted_starts)
        # only keep segments longer than min_count
        filtering = diff >= min_count
        predicted_segments = np.array(predicted_segments, dtype=object)[
            filtering].tolist()
        predicted_ends = np.array(predicted_ends)[filtering].tolist()
        predicted_starts = np.array(predicted_starts)[filtering].tolist()
    if merge_consecutive:
        # need to merge segments that are consecutive with synonyms
        to_merge = np.zeros(len(predicted_segments))
        for pred_idx, pred_segment in enumerate(predicted_segments):
            if pred_idx == len(predicted_segments) - 1:
                break
            next_segment = predicted_segments[pred_idx + 1]
            next_segment = extend(labels=next_segment, synonyms=synonyms)
            equality_check = any(
                [pred_lbl in next_segment for pred_lbl in pred_segment]
            )
            if equality_check:
                to_merge[pred_idx + 1] = 1
        # merge segments
        if to_merge.sum() > 0:
            segments_to_merge = np.where(to_merge == 1)[0]
            new_segments = []
            new_starts = []
            new_ends = []
            for segment_idx, segment in enumerate(predicted_segments):
                if segment_idx not in segments_to_merge:
                    new_segments.append(segment)
                    new_starts.append(predicted_starts[segment_idx])
                    new_ends.append(predicted_ends[segment_idx])
                else:
                    new_segments[-1].extend(segment)
                    new_ends[-1] = predicted_ends[segment_idx]
            predicted_segments = new_segments
            predicted_starts = new_starts
            predicted_ends = new_ends
    if stride > 1:
        predicted_starts = (np.array(predicted_starts) * stride).tolist()
        predicted_ends = (np.array(predicted_ends) * stride).tolist()

    if return_all:
        return predicted_segments, predicted_starts, predicted_ends
    return predicted_segments

def f_score(
    frame_wise_predictions: List[List[str]],
    frame_wise_gt_labels: List[List[str]],
    frame_wise_probs: Optional[List[float]] = None,
    bg_class: List[str] = ["no annotation"],
    synonyms: Optional[dict] = None,
    tau: float = 0.0,
    min_count: int = 1,
    overlap: float = 0.1,
    stride: int = 2,
):
    """
    Compute F-score over segments.

    Args:
        frame_wise_predictions (List[List[str]]): list of frame-wise predictions.
        frame_wise_gt_labels (List[List[str]]): list of frame-wise ground truth labels.
        frame_wise_probs (Optional[List[float]], optional): list of frame-wise probabilities.
            Defaults to None.
        bg_class (List[str], optional): list of all classes in frame_wise_labels to ignore.
        synonyms (Optional[dict], optional): synonym dictionary. Defaults to None.
        tau (float): threshold for predictions. Defaults to 0.0.
        min_count (int): minimum count for segmenting. Defaults to 1.
        overlap (float): overlap threshold for F-score. Defaults to 0.1.
        stride (int): striding factor for predictions. Defaults to 2.

    Returns:
        true_p: true positives
        false_p: false positives
        false_n: false negatives
    """
    gt_segments, gt_start, gt_end = get_labels_start_end_time(
        frame_wise_gt_labels,
        bg_class=bg_class,
        synonyms=synonyms,
    )
    predicted_segments, predicted_starts, predicted_ends = pred_filter(
        frame_wise_predictions,
        frame_wise_probs=frame_wise_probs,
        bg_class=bg_class,
        synonyms=synonyms,
        tau=tau,
        min_count=min_count,
        return_all=True,
        stride=stride,
    )

    true_p, false_p, hits = 0, 0, np.zeros(len(gt_segments))
    assert len(predicted_segments) == len(predicted_starts) and len(
        predicted_starts) == len(predicted_ends)
    for pred_segment, pred_start, pred_end in zip(
            predicted_segments,
            predicted_starts,
            predicted_ends,
    ):
        intersection = np.minimum(pred_end, gt_end) - \
            np.maximum(pred_start, gt_start)
        union = np.maximum(pred_end, gt_end) - np.minimum(pred_start, gt_start)

        check_list = []
        for gt_segment in gt_segments:
            # check that pred_segment == gt_segment
            # up to synonyms
            gt_segment = extend(labels=gt_segment, synonyms=synonyms)
            equality_check = any(
                [pred_lbl in gt_segment for pred_lbl in pred_segment])
            if equality_check:
                check_list.append(True)
            else:
                check_list.append(False)
        iou = (1.0 * intersection / union) * check_list
        # get the best scoring segment
        try:
            idx = np.array(iou).argmax()
        except ValueError:
            raise ValueError("IoU is empty")
        if iou[idx] >= overlap and not hits[idx]:
            true_p += 1
            hits[idx] = True
        else:
            false_p += 1
    false_n = len(gt_segments) - sum(hits)
    return float(true_p), float(false_p), float(false_n)


def segment_iou(
    predicted_segments: List[List[str]],
    gt_segments: List[List[str]],
    synonyms: Optional[dict] = None,
    return_recall: bool = False,
):
    """
    Computation of the IoU at segment-level

    Args:
        predicted_segments (List[str]): list of predicted segments
        gt_segments (List[str]): list of ground truth segments
        synonyms (Optional[dict], optional): synonym dictionary. Defaults to None.
        return_recall (bool): whether to return recall in addition to IoU. Defaults to False.

    Returns:
        iou (float): IoU score between predicted and ground truth segments
        recall (float): recall score between predicted and ground truth segments.
            Only returned if return_recall is True.
    """
    hits = np.zeros(len(gt_segments))
    false_postive_pred = 0
    for pred_segments in predicted_segments:
        check_list = []
        for gt_segment in gt_segments:
            # check that pred_segment == gt_segment
            # up to synonyms
            gt_segment = extend(labels=gt_segment, synonyms=synonyms)
            equality_check = any(
                [pred_lbl in gt_segment for pred_lbl in pred_segments]
            )
            if equality_check:
                check_list.append(True)
            else:
                check_list.append(False)
        # get locations where check_list is True
        hits[np.where(check_list)[0]] = True
        if not any(check_list):
            false_postive_pred += 1
    iou = sum(hits) / (len(gt_segments) + false_postive_pred)
    recall = sum(hits) / len(gt_segments)
    if return_recall:
        return iou, recall
    return iou
