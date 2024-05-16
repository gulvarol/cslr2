"""CSLR evaluation with frame-level paradigm."""
import json
import os
import pickle
from glob import glob
from typing import Optional

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tabulate import tabulate
from tqdm import tqdm

from misc.save_cslr_vis_timelines import plot_rectangles, create_rectangle_vis
from utils.cslr_metrics import (
    compute_frame_level_metrics, edit_score,
    f_score, get_labels_start_end_time, levenstein,
    pred_filter, segment_iou
)
from utils.frame_level_evaluation_dict import (
    combine_gt_pred_dict, gt_csvs_to_frame_level_gt,
    pred_pickles_to_frame_level_predictions
)
from utils.synonyms import fix_synonyms_dict


def do_search(
    combined_dict: dict,
    synonyms: Optional[dict],
    optimal_tau: Optional[float] = None,
    optimal_mc: Optional[int] = None,
) -> dict:
    """
    Perform hyper-parameter search.

    Args:
        combined_dict (dict): combined dictionary
        synonyms (Optional[dict], optional): synonym dictionary.
        optimal_tau (Optional[float], optional): optimal tau. Defaults to None.
        optimal_mc (Optional[int], optional): optimal mc. Defaults to None.

    Returns:
        scores_df (pd.DataFrame): dataframe of scores
    """
    print("Starting the search for optimal parameters on unseen CSLR sequences.")
    frame_wise_predictions = combined_dict["words"]
    frame_wise_probabilities = combined_dict["probs"]
    frame_wise_gt_labels = combined_dict["frame_ground_truth"]

    assert len(frame_wise_predictions) == len(frame_wise_gt_labels) and \
        len(frame_wise_gt_labels) == len(frame_wise_probabilities)
    if optimal_tau is None or optimal_mc is None:
        if optimal_tau is not None:
            taus = [optimal_tau]
        else:
            taus = [.0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
        if optimal_mc is not None:
            mcs = [optimal_mc]
        else:
            mcs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    overlaps = [0.1, 0.25, 0.5, 0.75, 0.9]
    scores_df = pd.DataFrame(
        columns=["tau", "mc", "edit_score", "wer", "len_gt"]
    )
    for tau in tqdm(taus, desc="Tau", leave=False):
        for min_c in tqdm(mcs, desc=f"Min count for tau = {tau}", leave=False):
            edit_scores = []
            gt_lens = []
            true_positives = {overlap: [] for overlap in overlaps}
            false_positives = {overlap: [] for overlap in overlaps}
            false_negatives = {overlap: [] for overlap in overlaps}
            for pred, prob, gt_label in zip(
                frame_wise_predictions,
                frame_wise_probabilities,
                frame_wise_gt_labels,
            ):
                score, len_gt = edit_score(
                    pred,
                    gt_label,
                    frame_wise_probs=prob,
                    synonyms=synonyms,
                    norm=False,
                    tau=tau,
                    min_count=min_c,
                )
                if len_gt > 0:
                    # only count if there is a ground truth
                    edit_scores.append(score)
                    gt_lens.append(len_gt)
                    # f1-score computation
                    for overlap in overlaps:
                        true_pos, false_pos, false_neg = f_score(
                            pred,
                            gt_label,
                            prob,
                            bg_class=["no annotation"],
                            synonyms=synonyms,
                            tau=tau,
                            min_count=min_c,
                            stride=2
                        )
                        true_positives[overlap].append(true_pos)
                        false_positives[overlap].append(false_pos)
                        false_negatives[overlap].append(false_neg)
            # print(f"Tau {tau}, MC: {min_c}, Sum: {np.sum(gt_lens)}")
            scores_dict = {
                "tau": [tau],
                "mc": [min_c],
                "edit_score": [np.mean(edit_scores)],
                "wer": [np.sum(edit_scores) / np.sum(gt_lens) * 100],
                "len_gt": [np.sum(gt_lens)],
            }
            for overlap in overlaps:
                f1_overlap = 2 * np.array(true_positives[overlap]) / \
                    (
                        2 * np.array(true_positives[overlap]) +
                        np.array(false_positives[overlap]) +  # neurips bug fix
                        # np.array(true_positives[overlap]) +
                        np.array(false_negatives[overlap])
                ) * 100
                scores_dict[f"f1@{overlap}"] = [np.mean(f1_overlap)]
            current_scores = pd.DataFrame.from_dict(scores_dict)
            scores_df = pd.concat(
                [scores_df, current_scores], ignore_index=True
            )
    return scores_df


def optimal_eval(
    combined_dict: dict,
    optimal_tau: float,
    optimal_mc: int,
    synonyms: Optional[dict],
    prediction_pickle_files: str,
    do_vis: bool = False,
    do_phrases_vis: bool = False,
    no_save: bool = False,
    effect_of_post_processing: bool = False,
    oracle: bool = False,
) -> None:
    """
    Perform final evaluation with optimal parameters.

    Args:
        combined_dict (dict): combined dictionary
        optimal_tau (float): optimal tau
        optimal_mc (int): optimal mc
        synonyms (Optional[dict], optional): synonym dictionary.
        prediction_pickle_files (str): path to prediction pickle files.
        do_vis (bool): whether to save visualisations or not. Defaults to False.
        do_phrases_vis (bool): whether to save visualisations for phrases or not.
        no_save (bool): whether to save scores or not. Defaults to False.
        effect_of_post_processing (bool): whether to save the effect of post-processing or not.
            Defaults to False.
        oracle (bool): whether to perform top-1 oracle filtering or not. Defaults to False.
    """
    results_df = pd.DataFrame(
        columns=[
            "episode_name", "sub",
            "gt", "pred",
            "wer", "edit_score",
            "f1@0.1", "f1@0.25", "f1@0.5", "f1@0.75", "f1@0.9"
        ]
    )
    edit_scores, gt_lens = [], []
    ious, recalls = [], []
    overlaps = [0.1, 0.25, 0.5, 0.75, 0.9]
    false_positives = {overlap: [] for overlap in overlaps}
    false_negatives = {overlap: [] for overlap in overlaps}
    true_positives = {overlap: [] for overlap in overlaps}
    for _, (pred, prob, gt_labels, ep_name, sub, s_time, e_time, raw_seg) in enumerate(
        zip(
            combined_dict["words"],
            combined_dict["probs"],
            combined_dict["frame_ground_truth"],
            combined_dict["episode_name_gt"],
            combined_dict["subtitles"],
            combined_dict["sub_start_gt"],
            combined_dict["sub_end_gt"],
            combined_dict["raw_segment_ground_truth"],
        )
    ):
        gt_segments, gt_starts, gt_ends = get_labels_start_end_time(
            gt_labels,
            bg_class=["no annotation"],
            synonyms=synonyms,
        )
        pred_segments, pred_starts, pred_ends = pred_filter(
            pred,
            prob,
            bg_class=["no annotation"],
            synonyms=synonyms,
            tau=optimal_tau,
            min_count=optimal_mc,
            merge_consecutive=False,
            return_all=True,
            stride=2,
        )
        if oracle:
            # will filter out all predictions that are not in the ground truth
            new_pred_segments, new_pred_starts, new_pred_ends = [], [], []
            for pred_segment, pred_start, pred_end in zip(
                pred_segments, pred_starts, pred_ends
            ):
                is_in_gt = False
                for gt_segment in gt_segments:
                    for annotation in gt_segment:
                        if synonyms is not None and annotation in synonyms.keys():
                            annotation = synonyms[annotation]
                        else:
                            annotation = [annotation]
                        local_in_gt = any(
                            [pred_lbl in annotation for pred_lbl in pred_segment]
                        )
                        if local_in_gt:
                            is_in_gt = True
                            break
                if is_in_gt:
                    new_pred_segments.append(pred_segment)
                    new_pred_starts.append(pred_start)
                    new_pred_ends.append(pred_end)
            pred_segments = new_pred_segments
            pred_starts = new_pred_starts
            pred_ends = new_pred_ends
        if len(gt_segments) > 0:
            # gt_segments could be empty if sign is too short in sub boundaries
            score, len_gt = levenstein(
                pred_segments,
                gt_segments,
                norm=False,
                synonyms=synonyms,
            )
            iou, recall = segment_iou(
                pred_segments,
                gt_segments,
                synonyms=synonyms,
                return_recall=True,
            )
            ious.append(iou)
            recalls.append(recall)
            if do_vis or do_phrases_vis:
                has_phrase = False
                for annotations in raw_seg:
                    for annotation in annotations:
                        nb_words = len(annotation.split())
                        if nb_words > 1:
                            has_phrase = True
                            break
                if do_vis or (do_phrases_vis and has_phrase):
                    gt_rectangles, pred_rectangles = create_rectangle_vis(
                        pred_segments,
                        pred_starts,
                        pred_ends,
                        gt_segments,
                        gt_starts,
                        gt_ends,
                        synonyms=synonyms,
                        stride=1,  # stride is already taken into account in pred_filter
                        effect_of_post_processing=effect_of_post_processing,
                    )
                    rectangles_1 = pred_rectangles
                    rectangles_2 = gt_rectangles
                    rectangles_3 = None
                    if effect_of_post_processing:
                        raw_segments, raw_starts, raw_ends = pred_filter(
                            pred, prob, bg_class=["no annotation"],
                            synonyms=synonyms, tau=0.0, min_count=1,
                            merge_consecutive=False, return_all=True,
                            stride=2,
                        )
                        _, raw_rectangles = create_rectangle_vis(
                            raw_segments,
                            raw_starts,
                            raw_ends,
                            gt_segments,
                            gt_starts,
                            gt_ends,
                            synonyms=synonyms,
                            stride=1,
                            remove_words=True,
                        )
                        rectangles_1 = raw_rectangles
                        rectangles_2 = pred_rectangles
                        rectangles_3 = gt_rectangles

                    save_path = os.path.join(
                        prediction_pickle_files, "visualisations_big_font_rotate"
                    )
                    if effect_of_post_processing:
                        save_path = os.path.join(
                            prediction_pickle_files,
                            "visualisations_big_font_rotate_effect_of_post_processing"
                        )
                    if not os.path.isdir(save_path):
                        print(f"Creating directory {save_path}")
                        os.makedirs(save_path)
                    plot_rectangles(
                        rectangles_1=rectangles_1,
                        rectangles_2=rectangles_2,
                        rectangles_3=rectangles_3,
                        subtitle=sub,
                        iou=iou,
                        wer=score / len_gt * 100,
                        episode_name=ep_name,
                        start_time=s_time,
                        end_time=e_time,
                        save_path=save_path,
                        fontsize=20,
                        diagonal=True,
                    )
            f1_scores = []
            for overlap in overlaps:
                true_pos, false_pos, false_neg = f_score(
                    pred,
                    gt_labels,
                    prob,
                    bg_class=["no annotation"],
                    synonyms=synonyms,
                    tau=optimal_tau,
                    min_count=optimal_mc,
                    overlap=overlap,
                )
                f1_scores.append(
                    2 * true_pos / (2 * true_pos +
                                    false_pos + false_neg) * 100
                )
                true_positives[overlap].append(true_pos)
                false_positives[overlap].append(false_pos)
                false_negatives[overlap].append(false_neg)
            results_df = pd.concat(
                [
                    results_df,
                    pd.DataFrame.from_dict(
                        {
                            "episode_name": [ep_name],
                            "subtitle": [sub],
                            "gt": [gt_segments],
                            "pred": [pred_segments],
                            "wer": [score / len_gt * 100],
                            "edit_score": [score],
                            "f1@0.1": [f1_scores[0]],
                            "f1@0.25": [f1_scores[1]],
                            "f1@0.5": [f1_scores[2]],
                            "f1@0.75": [f1_scores[3]],
                            "f1@0.9": [f1_scores[4]],
                            "iou": [iou],
                            "recall": [recall],
                        }
                    ),
                ],
                ignore_index=True,
            )
            edit_scores.append(score)
            gt_lens.append(len_gt)
    tau_list = [round(.1 * i, 2) for i in range(10)]
    precisions, recalls = [], []
    for tau in tau_list:
        prec, rec = compute_frame_level_metrics(
            combined_dict,
            synonyms=synonyms,
            tau=tau,
            do_print=False,
        )
        precisions.append(prec)
        recalls.append(rec)
    # final results dict
    final_results = {
        "optimal tau": optimal_tau,
        "optimal_mc": optimal_mc,
        "WER": np.sum(edit_scores) / np.sum(gt_lens) * 100,
        "Segment IoU": np.mean(ious) * 100,
        "Segment Recall": np.mean(recalls) * 100,
        "GT length": int(np.sum(gt_lens)),
        "Edit score": int(np.sum(edit_scores)),
    }
    for overlap in overlaps:
        f1_score = 2 * np.sum(true_positives[overlap]) / \
            (
                2 * np.sum(true_positives[overlap]) +
                np.sum(false_positives[overlap]) +
                np.sum(false_negatives[overlap])
        ) * 100
        final_results[f"mF1@{overlap}"] = f1_score
    final_results["frame accuracy"] = precisions[0] * 100

    # display + save scores
    table = [[key, value] for key, value in final_results.items()]
    print("\n")
    print(
        tabulate(
            table,
            headers=["metric", "value"],
            numalign="right",
            floatfmt=".2f"
        )
    )
    print("\n")
    if not no_save:
        # save results in a txt file
        save_path = os.path.join(prediction_pickle_files, "scores")
        if not os.path.isdir(save_path):
            print(f"Creating directory {save_path}")
            os.makedirs(save_path)
        score_file = "scores"
        with open(os.path.join(save_path, f"{score_file}.txt"), "w") as result_txt_f:
            result_txt_f.write(
                tabulate(
                    table,
                    headers=["metric", "value"],
                    numalign="right",
                    floatfmt=".2f",
                )
            )
        # dump results in a json file
        with open(os.path.join(save_path, f"{score_file}.json"), "w") as result_json_f:
            json.dump(final_results, result_json_f)


@hydra.main(version_base=None, config_path="config", config_name="cslr2_eval")
def main(cfg: Optional[DictConfig] = None) -> None:
    """Main function to evaluate with frame-level paradigm."""
    # synonyms handling
    syns = None
    if not cfg.remove_synonyms_handling:
        syns = pickle.load(open(cfg.paths.synonyms_pkl, "rb"))
        syns = fix_synonyms_dict(syns)

    # load vocabulary
    vocab_dict = pickle.load(open(cfg.paths.vocab_pkl, "rb"))
    if "words_to_id" in vocab_dict.keys():
        vocab_dict = vocab_dict["words_to_id"]

    id2word = {v: k for k, v in vocab_dict.items()}

    # load ground truth annotations
    if not cfg.test_search:
        train_set_csv_files = glob(os.path.join(
            cfg.gt_csv_root, "0/train/*.csv"))
        print(f"Found {len(train_set_csv_files)} train csv files.")
        train_set_gt_dict = gt_csvs_to_frame_level_gt(train_set_csv_files, fps=cfg.fps)
    test_set_csv_files = glob(os.path.join(cfg.gt_csv_root, "0/test/*.csv"))
    print(f"Found {len(test_set_csv_files)} test csv files.")
    test_set_gt_dict = gt_csvs_to_frame_level_gt(test_set_csv_files, fps=cfg.fps)

    assert cfg.prediction_pickle_files is not None or cfg.checkpoint is not None, "Either prediction_pickle_files or model_path must be set."  # pylint: disable=line-too-long
    if cfg.prediction_pickle_files is not None:
        assert os.path.isdir(cfg.prediction_pickle_files), \
            "prediction_pickle_files must be a directory."
        # no need to extract features again
    else:
        # extract features
        raise NotImplementedError("Feature extraction not implemented yet.")

    # load predictions
    pickle_files = glob(os.path.join(cfg.prediction_pickle_files, "*.pkl"))
    print(f"Found {len(pickle_files)} pickle files.")
    pred_dict = pred_pickles_to_frame_level_predictions(
        pickle_files, id2word, synonyms=syns,
        automatic_annotations=cfg.automatic_annotations,
        remove_synonym_grouping=cfg.remove_synonym_grouping,
    )
    if not cfg.test_search:
        # this assumes that args.prediction_pickle_files has files for both train and test sets
        combined_dict = combine_gt_pred_dict(
            train_set_gt_dict, pred_dict,
        )
    else:
        combined_dict = combine_gt_pred_dict(
            test_set_gt_dict, pred_dict,
        )

    # if optimal parameters have not been provided,
    # do the search on unseen sequences from train (if args.test_search is not set)
    # else optimal parameters are computed on test sequences
    if cfg.optimal_tau is None or cfg.optimal_mc is None:
        scores_df = do_search(
            combined_dict,
            synonyms=syns,
            optimal_tau=cfg.optimal_tau,
            optimal_mc=cfg.optimal_mc,
        )
        print(scores_df)
        # find optimal tau and mc combination for lowest wer
        optimal_tau_mc = scores_df.loc[scores_df["wer"].idxmin()]
        optimal_tau = optimal_tau_mc["tau"]
        optimal_mc = optimal_tau_mc["mc"]
        print(f"Optimal tau: {optimal_tau}, optimal mc: {optimal_mc}")
        print(f"WER: {optimal_tau_mc['wer']:.2f}%")
    else:
        optimal_tau = cfg.optimal_tau
        optimal_mc = cfg.optimal_mc

    # compute scores on test set with optimal parameters
    if not cfg.test_search:
        # compute scores on test set with optimal parameters
        combined_dict = combine_gt_pred_dict(
            test_set_gt_dict, pred_dict,
        )
        frame_wise_predictions = combined_dict["words"]
        frame_wise_probabilities = combined_dict["probs"]
        frame_wise_gt_labels = combined_dict["frame_ground_truth"]
        assert len(frame_wise_predictions) == len(frame_wise_gt_labels) and \
            len(frame_wise_gt_labels) == len(frame_wise_probabilities)

    optimal_eval(
        combined_dict,
        optimal_tau=optimal_tau,
        optimal_mc=optimal_mc,
        synonyms=syns,
        prediction_pickle_files=cfg.prediction_pickle_files,
        do_vis=cfg.do_vis,
        do_phrases_vis=cfg.do_phrases_vis,
        no_save=cfg.no_save,
        effect_of_post_processing=cfg.effect_of_post_processing,
        oracle=False,
    )


if __name__ == "__main__":
    # run main function
    main()
