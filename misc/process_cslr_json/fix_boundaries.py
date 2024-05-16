"""
Python script to fix the boundaries of subtitles in the CSLR dataset.

The fix is meant to change boundaries of subtitles such that all annotations that are
associated with it completely fall within (intersection == 1).
"""
import argparse
import glob
import os

import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_file",
        type=str,
        required=True,
        help="Path to the CSV file containing the subtitles and annotations.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    args = parser.parse_args()
    # need to remove "/" from the end of the path
    args.csv_file = args.csv_file.rstrip("/")
    names = os.path.basename(args.csv_file)

    splits = ["train", "val", "test"]
    total_changes = {
        "all": {"start": 0, "end": 0},
        "train": {"start": 0, "end": 0},
        "val": {"start": 0, "end": 0},
        "test": {"start": 0, "end": 0},
    }
    for split in splits:
        print(f"Processing {split} split...")
        all_files = glob.glob(f"{args.csv_file}/0/{split}/*.csv")
        print(f"Found {len(all_files)} files.")
        for csv_file in all_files:
            # open csv file in question
            gt_df = pd.read_csv(csv_file, delimiter=",")
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
                    gt_times, gt_annots = zip(*sorted(zip(gt_times, gt_annots)))
                    left_boundary.append(
                        f"{gt_annots[0].strip().replace(' ', '-')} {gt_times[0]}"
                    )
                    right_boundary.append(
                        f"{gt_annots[-1].strip().replace(' ', '-')} {gt_times[-1]}"
                    )
            gt_df["left_boundary"] = left_boundary
            gt_df["right_boundary"] = right_boundary

            # need to loop over all subtitles now
            starts = gt_df["start_sub"].tolist()
            ends = gt_df["end_sub"].tolist()
            subs = gt_df["english sentence"].tolist()
            assert (
                len(starts)
                == len(ends)
                == len(subs)
                == len(left_boundary)
                == len(right_boundary)
            )

            updated_starts, updated_ends = [], []
            for start, end, sub, left_bound, right_bound in zip(
                starts, ends, subs, left_boundary, right_boundary
            ):
                if isinstance(left_bound, float) or isinstance(right_bound, float):
                    continue
                left_bound = left_bound.split(" ")
                right_bound = right_bound.split(" ")
                left_bound = left_bound[1].split("-")[0]
                right_bound = right_bound[1].split("-")[-1]
                left_bound = float(left_bound)
                right_bound = float(right_bound)
                if left_bound < start:
                    total_changes["all"]["start"] += 1
                    total_changes[split]["start"] += 1
                    if args.debug:
                        print(f"Start: {left_bound} -> {start}")
                if right_bound > end:
                    total_changes["all"]["end"] += 1
                    total_changes[split]["end"] += 1
                    if args.debug:
                        print(f"End: {right_bound} -> {end}")
                updated_starts.append(min(start, left_bound))
                updated_ends.append(max(end, right_bound))

            gt_df["start_sub"] = updated_starts
            gt_df["end_sub"] = updated_ends
            # save the csv file to another location
            new_name = names[:-8] + "extended_boundaries_" + names[-8:]
            new_location = csv_file.replace(names, new_name)
            if not os.path.exists(os.path.dirname(new_location)):
                os.makedirs(os.path.dirname(new_location))
                print(f"Created directory {os.path.dirname(new_location)}")
            gt_df.to_csv(new_location, index=False)
    print(total_changes)
