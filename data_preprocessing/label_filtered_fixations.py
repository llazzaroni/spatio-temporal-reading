"""Create word-transition labels for the filtered fixation dataset.

The label on a row describes the transition *into* that fixation from the
previous retained fixation in the same (text, reader) scanpath.  The first
fixation in each scanpath has no preceding transition and receives a missing
label.

Run from the repository root:

    python -m data_preprocessing.label_filtered_fixations
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INPUT_NAME = "hp_augmented_meco_100_1000_1_10_model.csv"
OUTPUT_NAME = "hp_augmented_meco_100_1000_1_10_model_filtered_transition_labels.csv"


def add_transition_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Filter to word-aligned fixations and attach mutually exclusive labels."""
    required_columns = {"text", "reader", "fixid", "ianum_word", "line"}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    # The filtered corpus keeps only fixations that land on a word.  Sorting is
    # explicit so labels do not depend on the input CSV's row order.
    labelled = (
        frame.loc[frame["ianum_word"].notna()]
        .copy()
        .sort_values(["text", "reader", "fixid"], kind="stable")
    )

    # Word numbering is text-global; these values define the first word on a
    # displayed line and let us record whether a line return is canonical.
    line_first_words = (
        labelled[["text", "line", "ianum_word"]]
        .drop_duplicates()
        .groupby(["text", "line"])["ianum_word"]
        .min()
        .rename("line_first_word")
    )
    labelled = labelled.join(line_first_words, on=["text", "line"])

    scanpaths = labelled.groupby(["text", "reader"], sort=False)
    labelled["previous_ianum_word"] = scanpaths["ianum_word"].shift()
    labelled["previous_line"] = scanpaths["line"].shift()

    previous_word = labelled["previous_ianum_word"]
    previous_line = labelled["previous_line"]
    word = labelled["ianum_word"]
    line = labelled["line"]
    is_first_fixation = previous_word.isna()

    # A forward move to a later line is assigned line_return.  Many natural
    # scanpaths land after the literal first word of that line; the companion
    # boolean preserves the stricter 'next line, first word' definition.
    is_canonical_line_return = (
        (line == previous_line + 1) & (word == labelled["line_first_word"])
    )
    conditions = [
        (line > previous_line),
        (line == previous_line) & (word == previous_word + 1),
        (line == previous_line) & (word > previous_word + 1),
        word == previous_word,
        ((line == previous_line) & (word < previous_word)) | (line < previous_line),
    ]
    choices = ["line_return", "forward", "skip", "refixation", "regression"]

    labels = np.select(conditions, choices, default=None)
    labelled["transition_label"] = pd.array(labels, dtype="string")
    labelled.loc[is_first_fixation, "transition_label"] = pd.NA
    labelled["is_canonical_line_return"] = is_canonical_line_return.astype(bool)

    if labelled.loc[~is_first_fixation, "transition_label"].isna().any():
        raise AssertionError("A non-initial fixation did not receive a label.")

    return labelled.drop(columns=["line_first_word"])


def main() -> Path:
    input_path = DATA_DIR / INPUT_NAME
    output_path = DATA_DIR / OUTPUT_NAME
    labelled = add_transition_labels(pd.read_csv(input_path))
    labelled.to_csv(output_path, index=False)

    print(f"Wrote {len(labelled):,} word-aligned fixations to {output_path}")
    print(labelled["transition_label"].value_counts(dropna=False).to_string())
    print(
        "Canonical line returns: "
        f"{labelled['is_canonical_line_return'].sum():,}"
    )
    return output_path


if __name__ == "__main__":
    main()
