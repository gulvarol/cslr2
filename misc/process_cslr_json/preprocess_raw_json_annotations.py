"""
Script to preprocess gloss annotations from CSLR dataset in json format.
Annotations are assigned to subtitles that are saved along their corresponding gloss sequences.
This script also computes statistics about the annotations and the subtitles.
It will also plot histograms of the annotations and the subtitles (see CVPR24's submission supmat).
"""
# imports
import argparse
import csv
import glob
import json
import os
import re
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import webvtt
from tqdm import tqdm


def opts() -> argparse.ArgumentParser:
    """
    Function to parse the arguments.

    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the directory containing the raw json annotations.",
    )
    parser.add_argument(
        "--subs_dir",
        type=str,
        required=True,
        help="Path to the directory containing the subtitles.",
    )
    parser.add_argument(
        "--subset2episode",
        type=str,
        required=True,
        help="Path to the json file mapping each episode to its subset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path to the directory where preprocessed csv annotations will be saved. \
            If None, nothing will be saved.",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default=None,
        help="Path to the directory where the plots will be saved. \
            If None, nothing will be saved.",
    )
    parser.add_argument(
        "--misalignment_fix",
        action="store_true",
        help="Whether to fix the misalignment between the annotations and the subtitles.",
    )
    parser.add_argument(
        "--pad",
        type=int,
        default=10,
        help="Padding to avoid missing annotations.",
    )
    return parser.parse_args()


def convert_seconds_to_hms(seconds: float) -> str:
    """
    Convert seconds to hours, minutes, seconds.

    Args:
        seconds (float): seconds to convert.

    Returns:
        str: hours, minutes, seconds.
    """
    milliseconds = str(round((seconds % 1) * 1000)).zfill(3)
    return (
        "2000-01-01 "
        + time.strftime("%H:%M:%S", time.gmtime(seconds))
        + "."
        + milliseconds
    )


def get_subtitles(
    subtitles_path: str,
    start: float,
    end: float,
    pad: float = 10,
    plot: bool = False,
):
    """
    Function to get the subtitles that are within the boundaries of the annotations.

    Args:
        subtitles (str): path to the subtitle vtt file.
        start (float): start time of the annotation.
        end (float): end time of the annotation.
        pad (float, optional): padding to avoid missing annotations.
            Defaults to 10.
        plot (bool, optional): whether to save information relevant for plotting.
            Defaults to False.

    Returns:
        subtitles_df (pd.DataFrame): dataframe containing the subtitles
            within the boundaries of the annotations.
        subtitles_list_df (List[dict]): list of dictionaries containing
            information relevant for plotting (boundaries).
        plot_list_annots_df (List[dict]): list of dictionaries containing
            information relevant for plotting (text).
        subtitle_boundaries (List[str]): list of subtitle boundaries (for plotting).
    """
    csv_file = subtitles_path.endswith(".csv")
    pad = 0.1 if csv_file else pad
    subtitles_dict = {
        "start_time": [],  # start time of the subtitle in seconds
        "start_time2": [],  # start time of the subtitle (for plotting)
        "end_time": [],  # end time of the subtitle in seconds
        "end_time2": [],  # end time of the subtitle (for plotting)
        "subtitle": [],  # subtitle
    }
    min_start = start - pad
    max_end = end + pad
    subtitles_list_df = [] if plot else None
    plot_list_annots_df = [] if plot else None
    subtitle_boundaries = [] if plot else None
    plot_idx = 0
    if csv_file:
        ep_df = pd.read_csv(subtitles_path, delimiter=",")
        starts = ep_df["start sub (after alignement heuristic 1)"].tolist()
        ends = ep_df["end sub (after alignement heuristic 1)"].tolist()
        subs = ep_df["english sentence"].tolist()
        assert len(starts) == len(ends) == len(subs)
        for subtitle_start, subtitle_end, subtitle_text in zip(starts, ends, subs):
            if subtitle_start >= min_start and subtitle_end <= max_end:
                subtitles_dict["start_time"].append(subtitle_start)
                subtitles_dict["end_time"].append(subtitle_end)
                subtitles_dict["subtitle"].append(subtitle_text)
                if plot:
                    subtitle_middle = (subtitle_start + subtitle_end) / 2
                    subtitles_dict["start_time2"].append(
                        convert_seconds_to_hms(subtitle_start)
                    )
                    subtitles_dict["end_time2"].append(
                        convert_seconds_to_hms(subtitle_end)
                    )
                    subtitles_list_df.append(
                        dict(
                            Task="Subtitles",
                            start=subtitles_dict["start_time2"][-1],
                            end=subtitles_dict["end_time2"][-1],
                            color=hash(plot_idx),
                        )
                    )
                    plot_list_annots_df.append(
                        dict(
                            x=convert_seconds_to_hms(subtitle_middle),
                            y=0,
                            text=subtitle_text,
                            showarrow=False,
                            font=dict(color="black"),
                        )
                    )
                    plot_idx += 1
                    subtitle_boundaries.append(subtitles_dict["start_time2"][-1])
                    subtitle_boundaries.append(subtitles_dict["end_time2"][-1])
                else:
                    subtitles_dict["start_time2"].append(subtitle_start)
                    subtitles_dict["end_time2"].append(subtitle_end)
    else:
        # open subtitle file
        subtitles = webvtt.read(subtitles_path)
        for subtitle in subtitles:
            subtitle_start = subtitle.start_in_seconds
            subtitle_end = subtitle.end_in_seconds
            subtitle_text = subtitle.text
            if subtitle_start >= min_start and subtitle_end <= max_end:
                subtitles_dict["start_time"].append(subtitle_start)
                subtitles_dict["end_time"].append(subtitle_end)
                subtitles_dict["subtitle"].append(subtitle_text)
                if plot:
                    subtitle_middle = (subtitle_start + subtitle_end) / 2
                    subtitles_dict["start_time2"].append(
                        convert_seconds_to_hms(subtitle_start)
                    )
                    subtitles_dict["end_time2"].append(
                        convert_seconds_to_hms(subtitle_end)
                    )
                    subtitles_list_df.append(
                        dict(
                            Task="Subtitles",
                            start=subtitles_dict["start_time2"][-1],
                            end=subtitles_dict["end_time2"][-1],
                            color=hash(plot_idx),
                        )
                    )
                    plot_list_annots_df.append(
                        dict(
                            x=convert_seconds_to_hms(subtitle_middle),
                            y=0,
                            text=subtitle_text,
                            showarrow=False,
                            font=dict(color="black"),
                        )
                    )
                    plot_idx += 1
                    subtitle_boundaries.append(subtitles_dict["start_time2"][-1])
                    subtitle_boundaries.append(subtitles_dict["end_time2"][-1])
                else:
                    subtitles_dict["start_time2"].append(subtitle_start)
                    subtitles_dict["end_time2"].append(subtitle_end)
    subtitles_df = pd.DataFrame(subtitles_dict)
    # sort by start time
    subtitles_df = subtitles_df.sort_values(by=["start_time"])
    return subtitles_df, subtitles_list_df, plot_list_annots_df, subtitle_boundaries


def load_json_file(
    json_file: str,
):
    """
    Loads a json file and Returns only the relevant information.

    Args:
        json_file (str): path to the json file.

    Returns:
        episode_name (str): name of the episode.
        start (float): start time of the annotated block.
        end (float): end time of the annotated block.
        annotations (dict): dictionary containing the annotations.
    """
    with open(json_file, "rb") as json_f:
        json_data = json.load(json_f)
    assert len(json_data["file"]) == 1
    episode_name = json_data["file"]["1"]["fname"].replace(".mp4", "")
    start, end = (
        json_data["file"]["1"]["src"].split("#")[-1].replace("t=", "").split(",")
    )
    annotations = json_data["metadata"]
    return episode_name, start, end, annotations


def preprocess_annotation(
    annotation: str,
) -> str:
    """
    Preprocess an annotation.
    Will (remove typos), *P1/*P2, *T.

    Args:
        annotation (str): annotation to preprocess.

    Returns:
        str: preprocessed annotation.
    """
    words = [word for word in annotation.split() if word != "" and word != "OR"]
    words = [re.sub(r"[()\[\]]", "", word).strip() for word in words]
    annotation = "/".join(words)
    annotation = annotation.replace("-", " ")  # this could be a problem.

    has_star = "*" in annotation
    if has_star:
        all_words = annotation.split("/")
        # lowercase all words except for the ones with *
        all_words = [
            word.lower() if "*" not in word else word.upper() for word in all_words
        ]
        # remove duplicates
        all_words = list(set(all_words))
        # sort such that * is at the end
        all_words = sorted(all_words, key=lambda x: "*" in x)
        annotation = " ".join(all_words)
        if "*P" in all_words:
            # need to make the distinction between *P1 and *P2
            p1_list = ["me", "this", "i", "here", "in", "that", "you"]
            if any(
                [p1_word in word.split() for p1_word in p1_list for word in all_words]
            ):
                annotation = annotation.replace("*P", "*P1")
            elif all(["*" in word for word in all_words]):
                annotation = annotation.replace("*P", "*P1")
            else:
                annotation = annotation.replace("*P", "*P2")
        if any(["*T" in word for word in all_words]) and "*T" not in all_words:
            # group all the *T together
            annotation = (
                annotation.replace("*TA", "*T")
                .replace("*TB", "*T")
                .replace("*TC", "*T")
                .replace("*TD", "*T")
                .replace("*TE", "*T")
                .replace("**T", "*T")
            )
        if "*FS" in all_words or "~*FS" in all_words:
            annotation = annotation.replace("~*FS", "*F").replace("*FS", "*F")
        if "G*" in all_words:
            annotation = annotation.replace("G*", "*G")
        if "D*" in all_words:
            annotation = annotation.replace("D*", "*D")
        if "SHIP*D" in all_words:
            annotation = annotation.replace("SHIP*D", "ship *D")
        if "TONGUE*P" in all_words:
            annotation = annotation.replace("TONGUE*P", "tongue *P2")
        if "SAILING*D" in all_words:
            annotation = annotation.replace("SAILING*D", "sailing *D")
        if "AGO*TA" in all_words:
            # small hack since *TA has been transformed to *T in annotation
            # but not in all_words
            annotation = annotation.replace("AGO*T", "ago *T")
        if "THIS*P" in all_words:
            annotation = annotation.replace("THIS*P", "this *P1")
        if "HAY*F" in all_words:
            annotation = annotation.replace("HAY*F", "hay *F")
        if "TONY*F" in all_words:
            annotation = annotation.replace("TONY*F", "tony *F")
    return annotation


def process_annotations(
    annotations: dict,
    plot: bool = False,
):
    """
    Process annotations to get the start and end time of each annotation.

    Args:
        annotations (dict): dictionary containing the annotations.
        plot (bool, optional): whether to save information relevant for plotting. Defaults to False.

    Returns:
        annotation_dict (dict): dictionary containing the processed annotations.
        plot_list_df (pd.DataFrame): dataframe containing information relevant for plotting.
        plot_list_cslr_df (pd.DataFrame): dataframe containing information relevant for plotting.
    """
    annotation_dict = {
        "start_time": [],  # start time of the annotation in seconds
        "start_time2": [],  # start time of the annotation (for plotting)
        "end_time": [],  # end time of the annotation in seconds
        "end_time2": [],  # end time of the annotation (for plotting)
        "duration": [],  # duration of the annotation in seconds
        "annotation": [],  # annotation
        "assigned_subtitle": [],  # subtitle assigned to the annotation
    }
    plot_list_df = [] if plot else None
    plot_list_cslr_df = [] if plot else None
    plot_idx = 0
    for _, annotation in annotations.items():
        annotation_start_time, annotation_end_time = annotation["z"]
        annotation_gloss = annotation["av"]
        assert len(annotation_gloss) == 1
        try:
            annotation_gloss = annotation_gloss["1"]
        except KeyError:
            annotation_gloss = ""
        if annotation_gloss != "":
            annotation_gloss = preprocess_annotation(annotation_gloss)
            duration = annotation_end_time - annotation_start_time
            annotation_dict["start_time"].append(annotation_start_time)
            annotation_dict["end_time"].append(annotation_end_time)
            annotation_dict["duration"].append(duration)
            annotation_dict["annotation"].append(annotation_gloss)
            annotation_dict["assigned_subtitle"].append(-1)  # not assigned yet
            if plot:
                annotation_middle_time = (
                    annotation_start_time + annotation_end_time
                ) / 2
                annotation_dict["start_time2"].append(
                    convert_seconds_to_hms(annotation_start_time)
                )
                annotation_dict["end_time2"].append(
                    convert_seconds_to_hms(annotation_end_time)
                )
                mid = convert_seconds_to_hms(annotation_middle_time)
                plot_list_df.append(
                    dict(
                        Task="CSLR Annotations",
                        start=annotation_dict["start_time2"][-1],
                        end=annotation_dict["end_time2"][-1],
                        color=hash(plot_idx),
                    )
                )
                plot_list_cslr_df.append(
                    dict(
                        x=mid,
                        y=1,
                        text=annotation_gloss,
                        showarrow=False,
                        font=dict(color="black"),
                        textangle=45,
                    )
                )
                plot_idx += 1
            else:
                annotation_dict["start_time2"].append(annotation_start_time)
                annotation_dict["end_time2"].append(annotation_end_time)
    annotation_df = pd.DataFrame(annotation_dict)
    # sort by start time
    annotation_df = annotation_df.sort_values(by=["start_time"])
    return annotation_df, plot_list_df, plot_list_cslr_df


def assign_annotations(
    annotation_df: pd.DataFrame,
    subtitles_df: pd.DataFrame,
    plot: bool = False,
):
    """
    Function to assign the annotations to the corresponding subtitles.

    Args:
        annotation_df (pd.DataFrame): dataframe containing the annotations.
        subtitles_df (pd.DataFrame): dataframe containing the subtitles.
        plot (bool, optional): whether to save information relevant for plotting.
            Defaults to False.

    Returns:
        annotation_df (pd.DataFrame): dataframe containing annotations and subtitles.
        plot_list_cslr2_df (List[dict]): list of dictionaries with information for plots.
    """
    subtitle_starts = np.array(subtitles_df["start_time"].tolist())
    subtitle_ends = np.array(subtitles_df["end_time"].tolist())
    subtitle_texts = np.array(subtitles_df["subtitle"].tolist())
    plot_list_cslr2_df = [] if plot else None
    # loop over all the annotations
    for idx, row in annotation_df.iterrows():
        annotation_start = row["start_time"]
        annotation_end = row["end_time"]
        annotation_text = row["annotation"]
        if "*NS" not in annotation_text:
            # compute the distance between the annotation and all subtitles
            ious = np.minimum(annotation_end, subtitle_ends) - np.maximum(
                annotation_start, subtitle_starts
            )
            ious /= annotation_end - annotation_start
            # check if max iou is not 0
            if np.max(ious) > 0:
                assigned_sbutitle_idx = np.argmax(ious)  # ious == 0
            else:
                # find closest subtitle
                annot_mid_time = (annotation_start + annotation_end) / 2
                dist_to_start = np.abs(annot_mid_time - subtitle_starts)
                dist_to_end = np.abs(annot_mid_time - subtitle_ends)
                closest_start_idx = np.argmin(dist_to_start)
                closest_start_dist = dist_to_start[closest_start_idx]
                closest_end_idx = np.argmin(dist_to_end)
                closest_end_dist = dist_to_end[closest_end_idx]
                assigned_sbutitle_idx = (
                    closest_start_idx
                    if closest_start_dist < closest_end_dist
                    else closest_end_idx
                )
            row["assigned_subtitle"] = subtitle_texts[assigned_sbutitle_idx]
            if plot:
                plot_list_cslr2_df.append(
                    dict(
                        Task="CSLR Annotations",
                        start=row["start_time2"],
                        end=row["end_time2"],
                        color=hash(assigned_sbutitle_idx),
                    )
                )
        else:
            row["assigned_subtitle"] = ""
        # update the annotation dataframe
        annotation_df.loc[idx] = row
    return annotation_df, plot_list_cslr2_df


def handle_plot(
    subtitles_df: List[dict],
    cslr_df: List[dict],
    annots_df: List[dict],
    cslr_annots_df: List[dict],
    subtitle_boundaries: List[str],
):
    """
    Plots the annotations and subtitles.

    Args:
        subtitles_df (List[dict]): list of dictionaries containing
            information relevant for plotting (subtitles).
        cslr_df (List[dict]): list of dictionaries containing
            information relevant for plotting (annotations).
        subtitles_annots_df (List[dict]): list of dictionaries containing
            information relevant for plotting (text)
        cslr_annots_df (List[dict]): list of dictionaries containing
            information relevant for plotting (text).
        subtitle_boundaries (List[str]): list of subtitle boundaries (for plotting).

    Returns:
        figure (plotly.graph_objects.Figure): figure containing the plot.
    """
    plot_df = pd.concat(
        [
            pd.DataFrame(subtitles_df),
            pd.DataFrame(cslr_df),
        ]
    )
    annots_df.extend(cslr_annots_df)
    figure = px.timeline(
        plot_df,
        x_start="start",
        x_end="end",
        y="Task",
        color="color",
        width=20000,
        height=350,
        color_continuous_scale=sns.color_palette("pastel", n_colors=100, as_cmap=True),
    )
    figure.update_layout(
        margin=dict(t=2, b=2, l=2, r=2),
    )
    figure["layout"]["annotations"] = annots_df
    figure.update_layout(showlegend=False)
    figure.update(layout_coloraxis_showscale=False)
    figure.update_xaxes(dtick=1000, tickformat="%H:%M:%S")
    figure.update_yaxes(autorange="reversed")
    for date in subtitle_boundaries:
        figure.add_vline(x=date, line_width=1, line_dash="dash", line_color="black")
    return figure


if __name__ == "__main__":
    args = opts()
    do_plot = args.plot_dir is not None
    do_save = args.output_dir is not None

    if do_save:
        for split in ["train", "val", "test"]:
            dir_to_save = os.path.join(args.output_dir, "0", split)
            if os.path.exists(dir_to_save):
                print(f"Directory {dir_to_save} already exists.")
                # delete all the csv files in the directory
                print("Deleting the directory.")
                os.system(f"rm -r {dir_to_save}")
            print(f"Creating directory {dir_to_save}")
            os.makedirs(dir_to_save)

    # get the list of all the json files
    json_files = glob.glob(os.path.join(args.input_dir, "*.json"))
    print(f"Found {len(json_files)} json files.")

    # open file mapping each episode to its subset
    with open(args.subset2episode, "rb") as json_f:
        subset2episode = json.load(json_f)
    episode2subset = {}
    for key in subset2episode.keys():
        eps_split = subset2episode[key]
        for ep in eps_split:
            episode2subset[ep] = key

    counts = {
        "total": {},
        "train": {},
        "val": {},
        "test": {},
    }
    annots_duration = {
        "total": {},
        "train": {},
        "val": {},
        "test": {},
    }
    subtitles_duration = {
        "total": [],
        "train": [],
        "val": [],
        "test": [],
    }
    subtitles_words = {
        "total": 0,
        "train": 0,
        "val": 0,
        "test": 0,
    }
    vocab = {
        "total": set(),
        "train": set(),
        "val": set(),
        "test": set(),
    }
    subtitles_vocab = {
        "total": set(),
        "train": set(),
        "val": set(),
        "test": set(),
    }

    # loop over all the json files
    for j_file in tqdm(json_files):
        ep_name, batch_start, batch_end, annots = load_json_file(j_file)
        split = episode2subset[ep_name]
        # process annotations
        annot_df, plt_list_df, plt_list_cslr_df = process_annotations(
            annotations=annots,
            plot=do_plot,
        )

        # filter subtitles to only get subtitles that are within the boundaries of the annotations
        sub_path = (
            os.path.join(args.subs_dir, ep_name + ".vtt")
            if not args.misalignment_fix
            else os.path.join(args.subs_dir, f"0/{split}/{ep_name}.csv")
        )
        subs_df, subs_list_df, plt_list_annots_df, sub_boundaries = get_subtitles(
            subtitles_path=sub_path,
            start=annot_df["start_time"].min(),
            end=annot_df["end_time"].max(),
            pad=args.pad,
            plot=do_plot,
        )

        # for each annotation, find the corresponding subtitle
        annot_df, plt_list_cslr2_df = assign_annotations(
            annotation_df=annot_df,
            subtitles_df=subs_df,
            plot=do_plot,
        )
        # plot
        if do_plot:
            fig = handle_plot(
                subtitles_df=subs_list_df,
                cslr_df=plt_list_df,
                annots_df=plt_list_annots_df,
                cslr_annots_df=plt_list_cslr_df,
                subtitle_boundaries=sub_boundaries,
            )
            fig.write_image(
                os.path.join(
                    args.plot_dir, ep_name + f"_{batch_start}--{batch_end}_raw.png"
                )
            )
            fig = handle_plot(
                subtitles_df=subs_list_df,
                cslr_df=plt_list_cslr2_df,
                annots_df=plt_list_annots_df,
                cslr_annots_df=plt_list_cslr_df,
                subtitle_boundaries=sub_boundaries,
            )
            fig.write_image(
                os.path.join(
                    args.plot_dir,
                    ep_name + f"_{batch_start}--{batch_end}_assigned.png",
                )
            )
        # save
        if do_save:
            # save the annotations
            # loop over subtitles
            csv_to_save = os.path.join(args.output_dir, "0", split, ep_name + ".csv")
            already_exists = os.path.exists(csv_to_save)
            write_option = "w" if (not already_exists) else "a"
            with open(csv_to_save, write_option) as csv_f:
                writer = csv.writer(csv_f)
                if not already_exists:
                    writer.writerow(
                        [
                            "start_sub",
                            "end_sub",
                            "english sentence",
                            "approx gloss sequence",
                        ]
                    )
                for sub_idx, sub_row in subs_df.iterrows():
                    sub_start = sub_row["start_time"]
                    sub_end = sub_row["end_time"]
                    sub_text = sub_row["subtitle"]
                    subtitle_str = ""
                    # loop over annotations
                    for annot_idx, annot_row in annot_df.iterrows():
                        annot_start = annot_row["start_time"]
                        annot_end = annot_row["end_time"]
                        annot_text = annot_row["annotation"]
                        assigned_subtitle = annot_row["assigned_subtitle"]
                        if sub_text == assigned_subtitle:
                            # save the annotation to the corresponding subtitle
                            annot_words = "/".join(annot_text.split())
                            annot = f"{annot_words}[{annot_start}--{annot_end}]"
                            subtitle_str += annot
                            # compute statistics
                            for sp in ["total", split]:
                                try:
                                    counts[sp]["total"] += 1
                                except KeyError:
                                    counts[sp]["total"] = 1
                            # first check if the annotation is purely lexical
                            annot_has_star = "*" in annot_text
                            annot_all_words = annot_text.split()
                            for word in annot_all_words:
                                if "*" in word:
                                    for sp in ["total", split]:
                                        try:
                                            counts[sp][word] += 1
                                            annots_duration[sp][word].append(
                                                annot_end - annot_start
                                            )
                                        except KeyError:
                                            counts[sp][word] = 1
                                            annots_duration[sp][word] = [
                                                annot_end - annot_start
                                            ]
                                else:
                                    # update the vocabulary
                                    vocab["total"].add(word)
                                    vocab[split].add(word)
                            if not annot_has_star:
                                for sp in ["total", split]:
                                    try:
                                        counts[sp]["purely lexical"] += 1
                                        annots_duration[sp]["purely lexical"].append(
                                            annot_end - annot_start
                                        )
                                    except KeyError:
                                        counts[sp]["purely lexical"] = 1
                                        annots_duration[sp]["purely lexical"] = [
                                            annot_end - annot_start
                                        ]
                    if subtitle_str != "":
                        for sp in ["total", split]:
                            subtitles_duration[sp].append(sub_end - sub_start)
                            all_sub_words = sub_text.split()
                            # remove all punctuation
                            all_sub_words = (
                                re.sub(r'[()\[\].,;!?"\']', "", " ".join(all_sub_words))
                                .lower()
                                .split()
                            )
                            subtitles_words[sp] += len(all_sub_words)
                            for word in all_sub_words:
                                subtitles_vocab[sp].add(word)

                    # write to csv
                    if subtitle_str != "" and do_save:
                        writer.writerow(
                            [
                                sub_start,
                                sub_end,
                                sub_text,
                                subtitle_str.strip(),
                            ]
                        )

    # statistics print
    for split in ["train", "val", "test"]:
        if split in counts:
            print(f"Split: {split}")
            # order the counts by value
            counts[split] = {
                k: v
                for k, v in sorted(
                    counts[split].items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            }
            for key in counts[split].keys():
                print(f"{key}: {counts[split][key]}")
            print()

    for split in ["total", "train", "val", "test"]:
        # print number of subtitles + total duration
        msg = f"Number of subtitles in {split}: {len(subtitles_duration[split])}"
        msg += f" ({np.sum(subtitles_duration[split]) / 3600:.2f} hours)"
        print(msg)
        # print number of words in subtitles
        msg = f"Number of words in subtitles in {split}: {subtitles_words[split]}"
        msg += f" (vocab size: {len(subtitles_vocab[split])})"
        print(msg)
        # print number of annotations, annotation vocabulary size,
        # number of annotations that are not purely lexical
        total_count = 0
        for key in counts[split].keys():
            if key not in ["purely lexical", "total"]:
                total_count += counts[split][key]
        msg = f"Number of annotations in {split}: {counts[split]['total']}"
        msg += f" (vocab size: {len(vocab[split])},  sign type annotations: {total_count})"
        print(msg)

    print("Generating histograms")
    sns.set(style="darkgrid")
    save_path = os.path.join("misc/process_cslr_json", "plots")
    if not os.path.exists(save_path):
        print(f"Creating directory {save_path}")
        os.makedirs(save_path)
    # save the histogram for the duration of the annotationn for the test set
    # remove lexical annotations
    duration_dict = {
        "star sign type": [],
        "duration": [],
    }
    for key in annots_duration["test"].keys():
        if key in [
            "purely lexical",
            "*F",
            "*P1",
            "*P2",
            "*D",
            "*N",
            "*S",
            "*FE",
            "*G",
            "*U",
            "*T",
        ]:
            new_key = key
            if key == "purely lexical":
                new_key = "L"
            duration_dict["star sign type"].extend(
                [new_key] * len(annots_duration["test"][key])
            )
            duration_dict["duration"].extend(annots_duration["test"][key])
    duration_df = pd.DataFrame(duration_dict)
    histogram = sns.histplot(
        data=duration_df,
        x="duration",
        binrange=(0, 3),
        bins=30,
        binwidth=0.1,
    )
    histogram.set(xlabel="Duration (s)")
    histogram.figure.savefig(
        os.path.join(save_path, "cslr_test_anns_hist.png"),
        dpi=500,
    )
    # plot stacked histogram
    plt.clf()
    histogram = sns.kdeplot(
        data=duration_df,
        x="duration",
        hue="star sign type",
        palette="Set3",
        linewidth=1,
        clip=(0, 3),
        legend=True,
        common_norm=False,
    )
    for line in histogram.lines:
        line.set_ydata(line.get_ydata() / line.get_ydata().max())
    plt.ylim(0, 1.1)
    histogram.set(xlabel="Duration (s)")
    histogram.figure.savefig(
        os.path.join(save_path, "cslr_test_per_sign_density.png"),
        dpi=500,
    )
    # plot filled histogram
    plt.clf()
    histogram = sns.kdeplot(
        data=duration_df,
        x="duration",
        hue="star sign type",
        multiple="fill",
        palette="Set3",
        linewidth=0,
        clip=(0, 3),
        legend=True,
    )
    histogram.set(xlabel="Duration (s)")
    histogram.figure.savefig(
        os.path.join(save_path, "cslr_test_per_duration_density.png"),
        dpi=500,
    )
