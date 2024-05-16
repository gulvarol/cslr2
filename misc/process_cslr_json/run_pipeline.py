"""
Running full pipeline for CSLR data pre-processing.
"""
import os

from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to the directory containing CSLR json data.",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to the output directory.",
        required=True,
    )
    parser.add_argument(
        "--subs_dir",
        type=str,
        help="Path to the directory containing subs files (manually aligned).",
        required=True,
    )
    parser.add_argument(
        "--subset2episode",
        type=str,
        help="Path to json file containing subset to episode mapping.",
        required=True,
    )
    args = parser.parse_args()
    args.output_dir = args.output_dir.rstrip("/")
    # the following assumes that the output_directory last 8 characters are date in format DD.MM.YY
    output_dir2 = args.output_dir[:-8] + "extended_boundaries_" + args.output_dir[-8:]
    output_dir3 = output_dir2[:-8] + "fix_alignment_" + output_dir2[-8:]

    assignment_command = "python misc/process_cslr_json/preprocess_raw_json_annotations.py "
    assignment_command += f"--output_dir {args.output_dir} --input_dir {args.input_dir} "
    assignment_command += f"--subs_dir {args.subs_dir} --subset2episode {args.subset2episode}"

    # run assignment command
    print(f"Running command: {assignment_command}")
    os.system(assignment_command)

    fix_boundaries_command = "python misc/process_cslr_json/fix_boundaries.py "
    fix_boundaries_command += f"--csv_file {args.output_dir}"
    # run fix boundaries command
    print(f"Running command: {fix_boundaries_command}")
    os.system(fix_boundaries_command)

    fix_alignment_command = "python misc/process_cslr_json/fix_alignment.py "
    fix_alignment_command += f"--csv_file {output_dir2}"
    fix_alignment_command2 = "python misc/process_cslr_json/preprocess_raw_json_annotations.py "
    fix_alignment_command2 += f"--output_dir {output_dir3} --input_dir {args.input_dir} "
    fix_alignment_command2 += f"--subs_dir {output_dir2} --misalignment_fix "
    fix_alignment_command2 += f"--subset2episode {args.subset2episode}"
    # run fix alignment command
    print(f"Running command: {fix_alignment_command}")
    os.system(fix_alignment_command)
    print(f"Running command: {fix_alignment_command2}")
    os.system(fix_alignment_command2)

    remove_stars_command = "python misc/process_cslr_json/remove_star_annots_from_csv.py "
    remove_stars_command += f"--csv_root {output_dir2}"
    remove_stars_command2 = "python misc/process_cslr_json/remove_star_annots_from_csv.py "
    remove_stars_command2 += f"--csv_root {output_dir3}"
    # run remove stars command
    print(f"Running command: {remove_stars_command}")
    os.system(remove_stars_command)
    print(f"Running command: {remove_stars_command2}")
    os.system(remove_stars_command2)
