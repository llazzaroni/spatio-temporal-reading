"""Create geometry-aware behavioural labels for filtered fixations.

Labels describe transitions into each fixation from the preceding retained
fixation of the same (text, reader) scanpath.  The rules combine word/line
assignments with fixation geometry and retain ambiguity as ``uncertain``.

Run from the repository root:

    python -m data_preprocessing.label_filtered_fixations_geometry
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INPUT_NAME = "hp_augmented_meco_100_1000_1_10_model.csv"
TEXTS_NAME = "hp_eng_texts_100_1000_1_10.csv"
OUTPUT_NAME = "hp_augmented_meco_100_1000_1_10_model_filtered_behavioral_labels.csv"


def text_layout(texts: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return word ranks and vertical line centres from the text layout."""
    lines = texts.copy()
    lines["line_center_y"] = (lines["bbox_y1"] + lines["bbox_y2"]) / 2
    line_centres = lines.groupby(["text_id", "line"])["line_center_y"].median()

    words = (
        lines.loc[lines["ianum_word"].notna(), ["text_id", "line", "ianum_word"]]
        .drop_duplicates()
        .sort_values(["text_id", "line", "ianum_word"])
    )
    words["word_position_in_line"] = words.groupby(["text_id", "line"]).cumcount() + 1
    words["words_in_line"] = words.groupby(["text_id", "line"])["ianum_word"].transform("size")
    word_ranks = words.set_index(["text_id", "line", "ianum_word"])[
        ["word_position_in_line", "words_in_line"]
    ]

    line_spacing = line_centres.groupby(level=0).apply(
        lambda centres: np.median(np.abs(np.diff(np.sort(centres.to_numpy()))))
    )
    line_spacing.name = "line_spacing"
    return word_ranks, line_centres, line_spacing


def add_behavioural_labels(
    fixation_data: pd.DataFrame,
    texts: pd.DataFrame,
    vertical_threshold: float = 0.5,
    return_position_threshold: float = 0.5,
) -> pd.DataFrame:
    """Label filtered fixations while preserving geometry-quality diagnostics."""
    data = (
        fixation_data.loc[fixation_data["ianum_word"].notna()]
        .copy()
        .sort_values(["text", "reader", "fixid"], kind="stable")
    )
    word_ranks, line_centres, line_spacing = text_layout(texts)
    data = data.join(word_ranks, on=["text", "line", "ianum_word"])
    data = data.join(line_centres.rename("line_center_y"), on=["text", "line"])
    data = data.join(line_spacing, on="text")

    scanpaths = data.groupby(["text", "reader"], sort=False)
    shifted_columns = [
        "fixid", "ia_word", "ianum_word", "line", "x", "y", "line_center_y",
        "word_position_in_line", "words_in_line", "word_bbox_x1", "word_bbox_x2",
        "word_bbox_y1", "word_bbox_y2",
    ]
    for column in shifted_columns:
        data[f"previous_{column}"] = scanpaths[column].shift()

    data["word_difference"] = data["ianum_word"] - data["previous_ianum_word"]
    data["fixid_gap"] = data["fixid"] - data["previous_fixid"]
    data["is_consecutive_original_fixation"] = data["fixid_gap"].eq(1)
    data["line_difference"] = data["line"] - data["previous_line"]
    data["horizontal_displacement"] = data["x"] - data["previous_x"]
    data["vertical_displacement"] = data["y"] - data["previous_y"]
    data["expected_vertical_displacement"] = (
        data["line_center_y"] - data["previous_line_center_y"]
    )
    data["vertical_disagreement"] = (
        (data["vertical_displacement"] - data["expected_vertical_displacement"])
        .abs()
        / data["line_spacing"]
    )
    data["vertical_consistent"] = data["vertical_disagreement"] < vertical_threshold

    # A word alignment is reliable when its fixation is actually inside the
    # assigned word box.  Both fixations must satisfy this test for a labelled
    # transition; this is deliberately conservative.
    current_in_word_box = (
        data["x"].between(data["word_bbox_x1"], data["word_bbox_x2"])
        & data["y"].between(data["word_bbox_y1"], data["word_bbox_y2"])
    )
    previous_in_word_box = (
        data["previous_x"].between(data["previous_word_bbox_x1"], data["previous_word_bbox_x2"])
        & data["previous_y"].between(data["previous_word_bbox_y1"], data["previous_word_bbox_y2"])
    )
    data["current_word_assignment_reliable"] = current_in_word_box
    data["previous_word_assignment_reliable"] = previous_in_word_box
    data["transition_geometry_reliable"] = (
        data["vertical_consistent"] & current_in_word_box & previous_in_word_box
    )

    is_first = data["previous_ianum_word"].isna()
    same_line = data["line_difference"].eq(0)
    later_line = data["line_difference"].gt(0)
    earlier_line = data["line_difference"].lt(0)
    word_difference = data["word_difference"]
    reliable = data["transition_geometry_reliable"]
    previous_relative_position = (
        data["previous_word_position_in_line"] / data["previous_words_in_line"]
    )
    current_relative_position = data["word_position_in_line"] / data["words_in_line"]

    labels = pd.Series("uncertain", index=data.index, dtype="string")
    labels.loc[reliable & same_line & word_difference.eq(0)] = "refixation"
    labels.loc[
        reliable & same_line & word_difference.eq(1) & data["horizontal_displacement"].gt(0)
    ] = "forward"
    labels.loc[
        reliable & same_line & word_difference.gt(1) & data["horizontal_displacement"].gt(0)
    ] = "skip"
    labels.loc[
        reliable
        & ((earlier_line) | (same_line & word_difference.lt(0) & data["horizontal_displacement"].lt(0)))
    ] = "regression"

    # A normal left-to-right return sweep reaches the immediately next line,
    # moves leftward, and travels from a later to an earlier relative position.
    normal_return = (
        reliable
        & data["line_difference"].eq(1)
        & data["horizontal_displacement"].lt(0)
        & previous_relative_position.ge(return_position_threshold)
        & current_relative_position.le(return_position_threshold)
    )
    labels.loc[normal_return] = "line_return"
    labels.loc[reliable & later_line & ~normal_return] = "cross_line_other"
    labels.loc[is_first] = pd.NA
    data["behavioral_label"] = labels

    # Flag, rather than alter, local a→b→a line patterns with nearly monotonic
    # horizontal motion and little observed vertical movement.
    next_line = scanpaths["line"].shift(-1)
    next_x = scanpaths["x"].shift(-1)
    next_y = scanpaths["y"].shift(-1)
    data["local_line_assignment_flag"] = (
        data["previous_line"].eq(next_line)
        & data["line"].ne(data["previous_line"])
        & data["previous_x"].le(data["x"])
        & data["x"].le(next_x)
        & (pd.concat([data["previous_y"], data["y"], next_y], axis=1).max(axis=1)
           - pd.concat([data["previous_y"], data["y"], next_y], axis=1).min(axis=1)
           < vertical_threshold * data["line_spacing"])
    ).fillna(False)
    return data


def main() -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vertical-threshold", type=float, default=0.5)
    parser.add_argument("--return-position-threshold", type=float, default=0.5)
    args = parser.parse_args()

    output_path = DATA_DIR / OUTPUT_NAME
    labelled = add_behavioural_labels(
        pd.read_csv(DATA_DIR / INPUT_NAME),
        pd.read_csv(DATA_DIR / TEXTS_NAME),
        vertical_threshold=args.vertical_threshold,
        return_position_threshold=args.return_position_threshold,
    )
    labelled.to_csv(output_path, index=False)
    print(f"Wrote {len(labelled):,} rows to {output_path}")
    print(labelled["behavioral_label"].value_counts(dropna=False).to_string())
    print(f"Local line-assignment flags: {labelled['local_line_assignment_flag'].sum():,}")
    return output_path


if __name__ == "__main__":
    main()
