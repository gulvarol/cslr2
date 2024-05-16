"""Remove star annotations from CSLR csv files."""
import argparse
import os

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_root",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    save_dir = os.path.join(
        args.csv_root.replace("with_stars_", "")
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory {save_dir}")

    for split in ["train", "val", "test"]:
        csv_dir = os.path.join(args.csv_root, f"0/{split}")
        for csv_file in os.listdir(csv_dir):
            if csv_file.endswith(".csv"):
                # open file with pandas
                df = pd.read_csv(os.path.join(csv_dir, csv_file))

                # loop over rows
                for idx, row in df.iterrows():
                    # remove star annotations
                    if "*" in row["approx gloss sequence"]:
                        new_row = row["approx gloss sequence"]
                        new_row = new_row.replace("]", "[").split("[")
                        annots = new_row[::2]
                        times = new_row[1::2]
                        filtered_annots = []
                        filtered_times = []
                        for annot, time in zip(annots, times):
                            # check if there is a lexical annotation
                            has_lexical = any(
                                ["*" not in a for a in annot.split("/")])
                            if has_lexical:
                                annot = "/".join([
                                    a for a in annot.split("/")
                                    if "*" not in a
                                ])
                                filtered_annots.append(annot)
                                filtered_times.append(time)
                            else:
                                continue
                        if len(filtered_annots) == 0:
                            # can drop the row in question entirely
                            df.drop(idx, inplace=True)
                        else:
                            # replace the row with the filtered annotations
                            new_annots = ""
                            for annot, time in zip(filtered_annots, filtered_times):
                                new_annots += f"{annot}[{time}]"
                            df.loc[idx, "approx gloss sequence"] = new_annots
                # save the dataframe
                if not os.path.exists(os.path.join(save_dir, f"0/{split}")):
                    os.makedirs(os.path.join(save_dir, f"0/{split}"))
                    print(f"Created directory {save_dir}0/{split}")
                df.to_csv(
                    os.path.join(
                        save_dir,
                        f"0/{split}/{csv_file}"
                    ),
                    index=False
                )
