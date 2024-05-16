"""
Script to fix alignment of subtitles.

Heursitic implemented:
    - Assumption 1: glosses are well aligned.
        Only subtitles start and end times are to be modified.
    - For consecutive subtitles, look at boundary glosses:
        - If the boundary gloss better corresponds to the other subtitle,
            need to change start/end of the two subtitles.
        - If the boundary gloss better corresponds to the subtitle itself,
            no need to change.
        - If the boundary gloss is equally good for both subtitles,
            no need to change.

In the naive implementation, glosses are considered to correspond to a subtitle if
    the gloss word is in the subtitle text.
"""
import argparse
import glob
import os
import re
from typing import List


import pandas as pd
from nltk.stem import WordNetLemmatizer


def get_root_words(
    vocab_list: List[str],
) -> List[str]:
    """
    Get the root words from a vocabulary

    Args:
        vocab_list (List[str]): vocabulary of words

    Returns:
        root_words (List[str]): root words.
    """
    wordnet_lemmatizer = WordNetLemmatizer()
    root_words = [
        re.sub(
            r'[.,?!()"\']', '', word.lower().strip()
        ).split()
        for word in vocab_list
    ]
    root_words = [
        word for words in root_words for word in words if word != ""
    ]
    root_words = set(root_words)
    root_words = [
        wordnet_lemmatizer.lemmatize(
            wordnet_lemmatizer.lemmatize(word, pos="v"),
            pos="n"
        ) for word in root_words
    ]
    root_words = set(root_words)
    return root_words


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_file",
        type=str,
        required=True,
        help="Path to the csv file containing the subtitles."
    )
    parser.add_argument(
        "--do_star",
        action="store_true",
        help="If set, will save the star annotations as well."
    )
    parser.add_argument(
        "--max_shift",
        type=float,
        default=1.0,
        help="Maximum shift allowed for a subtitle (left or right), meaning it could be shifted by 2 * max_shift."  # pylint: disable=line-too-long
    )
    args = parser.parse_args()

    splits = ["train", "val", "test"]
    total_changes = {
        "all": {"start": 0, "end": 0, "double": 0, "bd_error": 0},
        "train": {"start": 0, "end": 0, "double": 0, "bd_error": 0},
        "val": {"start": 0, "end": 0, "double": 0, "bd_error": 0},
        "test": {"start": 0, "end": 0, "double": 0, "bd_error": 0},
    }
    for split in splits:
        print(f"Processing {split} split...")
        all_files = glob.glob(os.path.join(args.csv_file, f"0/{split}/*.csv"))
        print(f"Found {len(all_files)} files.")
        for csv_file in all_files:
            # open csv file in question
            gt_df = pd.read_csv(csv_file, delimiter=",")
            # hard copy
            gt_df = gt_df.copy()
            # sort by start_sub
            gt_df = gt_df.sort_values(by=["start_sub"], ascending=True)
            gt_glosses = gt_df["approx gloss sequence"].tolist()
            left_boundary = []
            right_boundary = []
            for gt_gloss in gt_glosses:
                if isinstance(gt_gloss, float):
                    # no gt
                    left_boundary.append("")
                    right_boundary.append("")
                else:
                    gt_gloss = gt_gloss.replace("]", "[").replace("'", "")
                    gt_annots = gt_gloss.split("[")[:-1]
                    gt_timings = gt_gloss.replace(" ", "/").replace("--", "-")
                    gt_timings = gt_timings.split("[")[:-1]
                    gt_times = gt_timings[1::2]
                    gt_annots = gt_annots[::2]
                    gt_times, gt_annots = zip(
                        *sorted(zip(gt_times, gt_annots))
                    )
                    left_boundary.append(
                        f"{gt_annots[0].strip().replace(' ', '-')} {gt_times[0]}"
                    )
                    right_boundary.append(
                        f"{gt_annots[-1].strip().replace(' ', '-')} {gt_times[-1]}"
                    )
            gt_df["left_boundary"] = left_boundary
            gt_df["right_boundary"] = right_boundary
            # need to loop by pairs of consecutive subtitles
            starts = gt_df["start_sub"].tolist()
            ends = gt_df["end_sub"].tolist()
            subs = gt_df["english sentence"].tolist()
            assert len(starts) == len(ends) and \
                len(ends) == len(subs) and \
                len(subs) == len(left_boundary) and \
                len(left_boundary) == len(right_boundary)

            # loop to change both start and times of subtitles
            # loop from left to right in timeline
            updated_starts = []
            updated_ends = []
            starts_changes = 0
            ends_changes = 0
            double_changes = 0  # i.e no change at all (heuristic)
            for i in range(len(starts) - 1):
                if i == 0:
                    # first subtitle start will not change
                    updated_starts.append(starts[i])

                first_sub = subs[i]
                second_sub = subs[i + 1]
                first_sub_end = ends[i]
                second_sub_start = starts[i + 1]
                if abs(second_sub_start - first_sub_end) < args.max_shift:
                    # maximum shift should be of args.max_shift seconds

                    # first check that the boundary word does not correspond to second subtitle
                    # meaning that the first subtitle is ending too late
                    # and that the second subtitle is starting too late as well
                    first_sub_right_boundary = right_boundary[i]
                    first_sub_words = get_root_words(first_sub.split(" "))
                    second_sub_words = get_root_words(second_sub.split(" "))
                    if first_sub_right_boundary in ["", " "]:
                        # no annotation
                        in_first_sub, in_second_sub = True, False
                    else:
                        try:
                            first_sub_right_boundary_words, first_sub_right_boundary_times = \
                                first_sub_right_boundary.split(" ")
                            first_sub_right_boundary_words = get_root_words(
                                first_sub_right_boundary_words.split("/")
                            )
                            in_first_sub, in_second_sub = False, False
                            for first_sub_right_boundary_word in first_sub_right_boundary_words:
                                if first_sub_right_boundary_word in first_sub_words:
                                    in_first_sub = True
                                if first_sub_right_boundary_word in second_sub_words:
                                    in_second_sub = True
                        except ValueError:
                            # no annotation ==> no change
                            in_first_sub, in_second_sub = True, False
                    first_change = False
                    if in_second_sub and not in_first_sub:
                        first_change = True
                        # means that the second subtitle should start earlier
                        # and first subtitle should end earlier
                        updated_starts.append(
                            float(
                                # to avoid same start and end times
                                first_sub_right_boundary_times.split("-")[0]
                            )
                        )
                        updated_ends.append(
                            float(
                                first_sub_right_boundary_times.split("-")[0]
                            ) - 1e-8
                        )
                        starts_changes += 1

                    second_sub_left_boundary = left_boundary[i + 1]
                    # second check that the boundary word does not correspond to first subtitle
                    # meaning that the second subtitle is starting too late
                    # and that the first subtitle is ending too early
                    if second_sub_left_boundary in ["", " "]:
                        in_first_sub, in_second_sub = False, True
                    else:
                        try:
                            second_sub_left_boundary_words, second_sub_left_boundary_times = \
                                second_sub_left_boundary.split(" ")
                            second_sub_left_boundary_words = get_root_words(
                                second_sub_left_boundary_words.split("/")
                            )
                            in_first_sub, in_second_sub = False, False
                            for second_sub_left_boundary_word in second_sub_left_boundary_words:
                                if second_sub_left_boundary_word in first_sub_words:
                                    in_first_sub = True
                                if second_sub_left_boundary_word in second_sub_words:
                                    in_second_sub = True
                        except ValueError:
                            # no annotation ==> no change
                            in_first_sub, in_second_sub = False, True
                    second_change = False
                    if in_first_sub and not in_second_sub:
                        second_change = True
                        if first_change:
                            # both changes are needed, so no change
                            updated_ends[-1] = first_sub_end
                            updated_starts[-1] = second_sub_start
                            double_changes += 1
                            starts_changes -= 1
                        else:
                            # means that the first subtitle should end later
                            # and second subtitle should start later
                            updated_ends.append(
                                float(
                                    second_sub_left_boundary_times.split(
                                        "-")[-1]
                                )
                            )
                            updated_starts.append(
                                float(
                                    second_sub_left_boundary_times.split(
                                        "-")[-1]
                                ) + 1e-8
                            )
                            ends_changes += 1

                    if not (first_change or second_change):
                        # no change
                        updated_starts.append(starts[i + 1])
                        updated_ends.append(ends[i])
                else:
                    # no change
                    updated_starts.append(starts[i + 1])
                    updated_ends.append(ends[i])

                if i == len(starts) - 2:
                    # last subtitle end will not change
                    updated_ends.append(
                        max(
                            ends[i + 1],
                            float(
                                second_sub_left_boundary_times.split("-")[-1]
                            )
                        )
                    )
            try:
                assert len(updated_starts) == len(updated_ends) and \
                    len(updated_ends) == len(starts)
            except AssertionError:
                pass

            # check that the new starts are smaller than the new ends
            for idx, (start, end) in enumerate(zip(updated_starts, updated_ends)):
                if start >= end:
                    print(
                        f"Error with {csv_file}: {subs[idx]} | {start} | {end}",
                        "Reverting to original start and end times."
                    )
                    updated_starts[idx] = starts[idx]
                    updated_ends[idx] = ends[idx]
                    for sp in ["all", split]:
                        total_changes[sp]["bd_error"] += 1

            gt_df["start sub (after alignement heuristic 1)"] = updated_starts
            gt_df["end sub (after alignement heuristic 1)"] = updated_ends
            # save the csv file
            gt_df.to_csv(csv_file, index=False)
            for sp in ["all", split]:
                total_changes[sp]["start"] += starts_changes
                total_changes[sp]["end"] += ends_changes
                total_changes[sp]["double"] += double_changes

            # open the star csv file in question
            if args.do_star:
                star_gt_df = pd.read_csv(
                    csv_file.replace(
                        "with_timings", "with_stars_with_timings2"),
                    delimiter=","
                )
                star_gt_df = star_gt_df.sort_values(
                    by=["start_sub"], ascending=True
                )
                star_gt_df["start sub (after alignement heuristic 1)"] = updated_starts
                star_gt_df["end sub (after alignement heuristic 1)"] = updated_ends
                star_gt_df.to_csv(
                    csv_file.replace(
                        "with_timings", "with_stars_with_timings2"
                    ),
                    index=False
                )
    print(total_changes)
