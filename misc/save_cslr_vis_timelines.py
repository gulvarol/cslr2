"""Python code to save CSLR timelines only."""
from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def plot_rectangles(
    rectangles_1: List,
    rectangles_2: List,
    rectangles_3: Optional[List],
    subtitle: str,
    iou: float,
    wer: float,
    episode_name: str,
    start_time: float,
    end_time: float,
    save_path: str,
    dpi: int = 500,
    fontsize: int = 8,
    default_y0: float = 0.25,
    default_y1: float = 1.25,
    default_y3: float = 2.25,
    default_height: float = 0.75,
    diagonal: bool = False,
) -> None:
    """
    Plot cslr visualisation timeline.

    Args:
        rectangles_1 (List): List of tuples containing the rectangles for the first line
            each rectangle is a tuple of (x, y, width, height, idx_c, text)
        rectangles_2 (List): List of tuples containing the rectangles for the second line
            each rectangle is a tuple of (x, y, width, height, idx_c, text)
        rectangles_3 (Optional[List]): List of tuples containing the rectangles for the third line
            if None, no third line is plotted
            each rectangle is a tuple of (x, y, width, height, idx_c, text)
        subtitle (str): subtitle of the plot
        iou (float): IoU of the prediction
        wer (float): WER of the prediction
        episode_name (str): episode name
        start_time (float): start time of the prediction
        end_time (float): end time of the prediction
        save_path (str): path to save the plot
        dpi (int, optional): dpi of the plot. Defaults to 500.
        fontsize (int, optional): fontsize of the plot. Defaults to 8.
        default_y0 (float, optional): y value of the first line. Defaults to 0.25.
        default_y1 (float, optional): y value of the second line. Defaults to 1.25.
        default_y3 (float, optional): y value of the third line. Defaults to 2.25.
        diagonal (bool, optional): whether to write the text in diagonal. Defaults to False.
    """
    rectangle_pad = 0.8 if diagonal else 0.0
    max_fontsize = 20
    sns.set(style="whitegrid", palette="pastel")
    colors = sns.color_palette("pastel", n_colors=100)
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    matplotlib.rcParams["font.family"] = "STIXGeneral"
    matplotlib.rcParams["text.usetex"] = False
    # split the subtitle into multiple lines (max 100 characters per line)
    sub_words = subtitle.split(" ")
    char_count = 0
    for i, word in enumerate(sub_words):
        char_count += len(word)
        if char_count > 75:
            sub_words.insert(i, "\n")
            char_count = 0
    subtitle = " ".join(sub_words)
    # remove the last \n if it exists
    subtitle = subtitle[:-1] if subtitle[-1] == "\n" else subtitle
    subtitle = subtitle + "\n" + \
        r"$\bf{WER:" + f"{wer:.2f}" + "}$ " + \
        r"$\bf{IoU:" + f"{iou:.2f}" + "}$"
    plt.clf()
    fig, ax = plt.subplots()
    plt.subplots_adjust(
        left=.1, bottom=.1, right=.9,
        top=.9, wspace=.2, hspace=1.8
    )
    fig.suptitle(subtitle, fontsize=max(max_fontsize, fontsize))
    fig.set_figheight(3)
    # set width of each subplot as 8
    fig.set_figwidth(12)
    min_val = 10000000
    max_val = 0
    y_values = {}
    for i, rectangle in enumerate(rectangles_1):
        x, y, width, height, idx_c, text = rectangle
        height = height + rectangle_pad
        y_values["rectangle1"] = {"y": y, "height": height}
        idx_c = idx_c % len(colors)
        color = colors[idx_c]

        if x < min_val:
            min_val = x
        if x + width > max_val:
            max_val = x + width

        rect = plt.Rectangle(
            (x, y), width, height, color=color
        )
        ax.add_patch(rect)

        if "/" in text:
            texts = text.split("/")
            new_text = ""
            nb_changes = 0
            for text in texts:
                if text not in new_text and len(text) > 1:
                    new_text += text + "\n"
                    nb_changes += 1
            text = new_text if nb_changes > 1 else \
                new_text.replace("\n", "")
            # this is ok when fontsize is 8 (might change otherwise)
            y = y - 0.12 if nb_changes > 1 else y
        ax.text(
            x + width/2, y + height/2, text,
            ha="center", va="center", color="black",
            fontsize=fontsize, rotation=45 if diagonal else 0,
        )
    for i, rectangle in enumerate(rectangles_2):
        x, y, width, height, idx_c, text = rectangle
        y += rectangle_pad
        height += rectangle_pad
        y_values["rectangle2"] = {"y": y, "height": height}
        idx_c = idx_c % len(colors)
        color = colors[idx_c]

        if x < min_val:
            min_val = x
        if x + width > max_val:
            max_val = x + width

        rect = plt.Rectangle((x, y), width, height, color=color)
        ax.add_patch(rect)

        if "/" in text:
            texts = text.split("/")
            new_text = ""
            nb_changes = 0
            for text in texts:
                if text not in new_text and len(text) > 1:
                    new_text += text + "\n"
                    nb_changes += 1
            text = new_text if nb_changes > 1 else \
                new_text.replace("\n", "")
            # this is ok when fontsize is 8 (might change otherwise)
            y = y - 0.12 if nb_changes > 1 else y
        ax.text(
            x + width/2, y + height/2, text,
            ha="center", va="center", color="black",
            fontsize=fontsize, rotation=45 if diagonal else 0,
        )
    if rectangles_3 is not None:
        for i, rectangle in enumerate(rectangles_3):
            x, y, width, height, idx_c, text = rectangle
            y += 2 * rectangle_pad
            height += rectangle_pad
            y_values["rectangle3"] = {"y": y, "height": height}
            idx_c = idx_c % len(colors)
            color = colors[idx_c]

            if x < min_val:
                min_val = x
            if x + width > max_val:
                max_val = x + width

            rect = plt.Rectangle((x, y), width, height, color=color)
            ax.add_patch(rect)

            if "/" in text:
                texts = text.split("/")
                new_text = ""
                nb_changes = 0
                for text in texts:
                    if text not in new_text and len(text) > 1:
                        new_text += text + "\n"
                        nb_changes += 1
                text = new_text if nb_changes > 1 else \
                    new_text.replace("\n", "")
                # this is ok when fontsize is 8 (might change otherwise)
                y = y - 0.12 if nb_changes > 1 else y
            ax.text(
                x + width/2, y + height/2, text,
                ha="center", va="center", color="black",
                fontsize=fontsize, rotation=45 if diagonal else 0,
            )

    ax.set_xlim(min_val - 1, max_val + 1)
    if rectangles_3 is not None:
        try:
            ax.set_yticks([])
            ax.set_ylim(
                0,
                y_values["rectangle3"]["y"] +
                y_values["rectangle3"]["height"] + y_values["rectangle1"]["y"]
            )
        except KeyError:
            ax.set_yticks(
                ticks=[
                    default_y0 + default_height / 2,
                    default_y1 + default_height / 2,
                    default_y3 + default_height / 2,
                ],
                labels=[
                    r"$\bf{Ours}",
                    r"$\bf{GT (eval)}$",
                    r"$\bf{GT (raw)}$"
                ],
                weight="bold",
                fontsize=max(max_fontsize, fontsize)
            )
            ax.set_ylim(
                0, default_y0 + default_y3 + default_height
            )
    else:
        try:
            ax.set_yticks([])
            ax.set_ylim(
                0,
                y_values["rectangle2"]["y"] +
                y_values["rectangle2"]["height"] + y_values["rectangle1"]["y"]
            )
        except KeyError:
            ax.set_yticks(
                ticks=[
                    default_y0 + default_height / 2,
                    default_y1 + default_height / 2
                ],
                labels=[r"$\bf{Ours}$", r"$\bf{GT (eval)}$"],
                weight="bold",
                fontsize=max(max_fontsize, fontsize)
            )
            ax.set_ylim(
                0, default_y0 + default_y1 + default_height
            )
    ax.grid(False)
    ax.set_xticks([])  # Hide y-axis ticks
    ax.get_xaxis().set_visible(False)
    fig.tight_layout()

    # save figure
    plt.savefig(
        f"{save_path}/{episode_name}_{int(start_time * 25)}_{int(end_time * 25)}_{int(wer)}.png",
        dpi=dpi,
    )
    plt.close()


def plot_rectangle(
    episode_name: str,
    begin_time: float,
    end_time: float,
    subtitle: str,
    rectangles: List,
    dpi: int = 500,
    save_root: str = "visualisations",
    width: int = 20,
) -> None:
    """
    Assemble rectangles into a matplotlib figure and save it.

    Args:
        episode_name (str): episode name
        begin_time (float): begin time of the rectangle
        end_time (float): end time of the rectangle
        subtitle (str): subtitle corresponding to the rectangle
        rectangles (List): list of rectangles with CSLR annotations/predictions
        dpi (int, optional): dpi of the plot. Defaults to 500.
        save_root (str, optional): root to save the visualisations. Defaults to "visualisations".
        width (int, optional): width of the plot. Defaults to 20.
    """
    plt.clf()
    fig, ax = plt.subplots()
    fig.set_figheight(5)
    fig.set_figwidth(width)
    ax.set_xticks([])
    ax.set_yticks([])

    sns.set(style="whitegrid", palette="pastel")
    colors = sns.color_palette("pastel", n_colors=100)
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    matplotlib.rcParams["font.family"] = "STIXGeneral"
    matplotlib.rcParams["text.usetex"] = False

    local_min = 10000000
    local_max = 0
    for i, rectangle in enumerate(rectangles):
        x, y, width, height, idx_c, text = rectangle
        if x < local_min:
            local_min = x
        if x + width > local_max:
            local_max = x + width
        rect = plt.Rectangle((x, y), width, height / 2, color=colors[idx_c])
        ax.add_patch(rect)
        # remove redundant text + put in capital letter when *
        words = text.split("/")
        words = [word if "*" not in word else word.upper() for word in words]
        # sort words such that * at the end
        print(words)
        print(words)
        text = " ".join(words)
        ax.text(
            x + width/2, y + height/4, text,
            ha="center", va="center", color="black",
            fontsize=30, rotation=0
        )
    sub_words = subtitle.split(" ")
    char_count = 0
    for i, word in enumerate(sub_words):
        char_count += len(word)
        if char_count > 75:
            sub_words.insert(i, "\n")
            char_count = 0
    subtitle = " ".join(sub_words)
    # remove the last \n if it exists
    subtitle = subtitle[:-1] if subtitle[-1] == "\n" else subtitle
    plt.suptitle(subtitle)
    fig.tight_layout()
    # ax.set_xlim(begin_time, end_time)
    ax.set_xlim(local_min - 1, local_max + 1)
    ax.get_xaxis().set_visible(False)
    ax.set_ylim(0, 1)
    print(
        f"Saving {episode_name}_{int(begin_time * 25)}_{int(end_time * 25)}.png")
    plt.savefig(
        f"{save_root}/{episode_name}_{int(begin_time * 25)}_{int(end_time * 25)}.png",
        dpi=dpi,
    )


def create_rectangle_vis(
    predicted_segments: List[List[str]],
    predicted_starts: List[int],
    predicted_ends: List[int],
    gt_segments: List[List[str]],
    gt_starts: List[int],
    gt_ends: List[int],
    synonyms: Optional[dict] = None,
    stride: int = 2,
    effect_of_post_processing: bool = False,
    remove_words: bool = False,
    only_one: bool = False,
):
    """
    Create a rectangle visualisation of the predictions and ground truth.

    Args:
        predicted_segments (List[List[str]]): list of predicted segments
        predicted_starts (List[int]): list of start times of the predicted segments
        predicted_ends (List[int]): list of end times of the predicted segments
        gt_segments (List[List[str]]): list of ground truth segments
        gt_starts (List[int]): list of start times of the ground truth segments
        gt_ends (List[int]): list of end times of the ground truth segments
        synonyms (Optional[dict], optional): synonym dictionary. Defaults to None.
        stride (int): striding factor for predictions. Defaults to 2.
        effect_of_post_processing (bool): whether to show the effect of post-processing.
        remove_words (bool): whether to remove words in the rectangles. Defaults to False.
        only_one (bool): whether to only show one set of rectangles. Defaults to False.

    Returns:
        gt_rectangles (List): list of ground truth rectangles, each rectangle with the format
            (x, y, width, height, idx_c, text)
        pred_rectangles (List): list of predicted rectangles, each rectangle with the format
            (x, y, width, height, idx_c, text)
    """
    offset = 1 if effect_of_post_processing else 0
    gt_rectangles, pred_rectangles = [], []
    all_words = {}
    for gt_segment in gt_segments:
        for word in gt_segment:
            syn_colored = False
            if synonyms is not None and word in synonyms.keys():
                syns = synonyms[word]
                for syn in syns:
                    if syn in all_words:
                        all_words[word] = all_words[syn]
                        syn_colored = True
                        break
            if not syn_colored and word not in all_words:
                all_words[word] = len(list(all_words.keys()))
    for pred_segment in predicted_segments:
        for word in pred_segment:
            syn_colored = False
            if synonyms is not None and word in synonyms.keys():
                syns = synonyms[word]
                for syn in syns:
                    if syn in all_words:
                        all_words[word] = all_words[syn]
                        syn_colored = True
                        break
            if not syn_colored and word not in all_words:
                all_words[word] = len(list(all_words.keys()))

    for gt_segment, gt_start, gt_end in zip(gt_segments, gt_starts, gt_ends):
        gt_rectangles.append(
            (
                gt_start,
                1.25 + offset if not only_one else 0.25 + offset,
                gt_end - gt_start,
                0.75 if not only_one else 0.5,
                all_words[gt_segment[0]],
                "/".join(gt_segment),
            )
        )
    for pred_segment, pred_start, pred_end in zip(
        predicted_segments,
        predicted_starts,
        predicted_ends
    ):
        pred_rectangles.append(
            (
                stride * pred_start,
                0.25 + offset if not only_one else 1.25 + offset,
                stride * (pred_end - pred_start),
                0.75 if not only_one else 0.5,
                all_words[pred_segment[0]],
                "/".join(pred_segment) if not remove_words else "",
            )
        )
    return gt_rectangles, pred_rectangles
