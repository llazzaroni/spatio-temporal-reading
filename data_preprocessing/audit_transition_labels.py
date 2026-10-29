"""Audit transition labels in the filtered word-level fixation dataset.

Run from the repository root:

    python -m data_preprocessing.audit_transition_labels

The script prints:
  1. skip-event word-index differences;
  2. line-return positions within the previous and current lines; and
  3. deterministic random examples from each transition category.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = (
    ROOT
    / "data"
    / "hp_augmented_meco_100_1000_1_10_model_filtered_transition_labels.csv"
)
DEFAULT_LABELS = ["forward", "skip", "refixation", "regression", "line_return"]


def add_audit_columns(frame: pd.DataFrame, label_column: str) -> pd.DataFrame:
    """Add previous-fixation fields and word ranks within displayed lines."""
    required = {
        "text", "reader", "fixid", "line", "ianum_word", "ia_word",
        "x", "y", label_column,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    data = frame.sort_values(["text", "reader", "fixid"], kind="stable").copy()
    has_layout_ranks = {"word_position_in_line", "words_in_line"}.issubset(data.columns)
    # The geometry-aware CSV already carries some of these diagnostics.  Rebuild
    # them here so this audit has one consistent definition for either CSV.
    derived_columns = [
        "word_difference", "line_difference",
        "previous_word", "previous_x", "previous_y", "previous_word_position_in_line",
        "previous_words_in_line", "previous_ia_word", "previous_line", "previous_ianum_word",
    ]
    if not has_layout_ranks:
        derived_columns.extend(["word_position_in_line", "words_in_line"])
    data = data.drop(columns=[column for column in derived_columns if column in data], errors="ignore")
    if not has_layout_ranks:
        word_layout = (
            data[["text", "line", "ianum_word"]]
            .drop_duplicates()
            .sort_values(["text", "line", "ianum_word"])
        )
        word_layout["word_position_in_line"] = (
            word_layout.groupby(["text", "line"]).cumcount() + 1
        )
        word_layout["words_in_line"] = word_layout.groupby(
            ["text", "line"]
        )["ianum_word"].transform("size")
        ranks = word_layout.set_index(["text", "line", "ianum_word"])[
            ["word_position_in_line", "words_in_line"]
        ]
        data = data.join(ranks, on=["text", "line", "ianum_word"])

    scanpaths = data.groupby(["text", "reader"], sort=False)
    for column in [
        "ia_word", "line", "ianum_word", "x", "y",
        "word_position_in_line", "words_in_line",
    ]:
        data[f"previous_{column}"] = scanpaths[column].shift()

    data["word_difference"] = data["ianum_word"] - data["previous_ianum_word"]
    data["line_difference"] = data["line"] - data["previous_line"]
    return data


def print_skip_distribution(data: pd.DataFrame, label_column: str) -> None:
    skips = data.loc[data[label_column].eq("skip"), "word_difference"]
    print("CHECK 1 — Skip word-index differences")
    print(skips.describe(percentiles=[.01, .05, .10, .25, .50, .75, .90, .95, .99]).to_string())
    print("\nCounts by difference:")
    print(skips.value_counts().sort_index().to_string())


def print_line_return_audit(data: pd.DataFrame, label_column: str) -> None:
    returns = data.loc[data[label_column].eq("line_return")].copy()
    print("\nCHECK 2 — Line-return positions")
    print("Line differences:")
    print(returns["line_difference"].value_counts().sort_index().to_string())
    print("\nPositions within the previous and current line:")
    print(
        returns[
            [
                "previous_word_position_in_line", "previous_words_in_line",
                "word_position_in_line", "words_in_line",
            ]
        ].describe(percentiles=[.10, .25, .50, .75, .90]).to_string()
    )

    late_to_early = (
        (returns["previous_word_position_in_line"] / returns["previous_words_in_line"] >= .5)
        & (returns["word_position_in_line"] / returns["words_in_line"] <= .5)
    )
    canonical = (
        returns["line_difference"].eq(1)
        & returns["word_position_in_line"].eq(1)
    )
    print(
        f"\nLate-half → early-half: {late_to_early.sum():,} / {len(returns):,} "
        f"({late_to_early.mean():.1%})"
    )
    print(
        f"Exact next-line, first-word returns: {canonical.sum():,} / {len(returns):,} "
        f"({canonical.mean():.1%})"
    )


def print_samples(data: pd.DataFrame, label_column: str, samples: int, seed: int) -> None:
    columns = [
        "text", "reader", "fixid", "previous_ia_word", "ia_word",
        "previous_line", "line", "previous_word_position_in_line",
        "previous_words_in_line", "word_position_in_line", "words_in_line",
        "previous_x", "previous_y", "x", "y", "word_difference",
        "line_difference", label_column,
    ]
    optional_columns = [
        "vertical_disagreement", "transition_geometry_reliable",
        "local_line_assignment_flag", "is_canonical_line_return",
    ]
    columns.extend(column for column in optional_columns if column in data.columns)
    labels = [label for label in DEFAULT_LABELS if label in set(data[label_column].dropna())]
    labels.extend(
        label for label in ["cross_line_other", "uncertain"]
        if label in set(data[label_column].dropna())
    )
    print(f"\nCHECK 3 — {samples} deterministic random samples per label (seed={seed})")
    for label in labels:
        subset = data.loc[data[label_column].eq(label), columns]
        n = min(samples, len(subset))
        print(f"\n[{label.upper()}]")
        print(subset.sample(n=n, random_state=seed).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--label-column",
        help="Label column to audit. Defaults to behavioral_label when available, otherwise transition_label.",
    )
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261029)
    args = parser.parse_args()

    raw_data = pd.read_csv(args.data)
    label_column = args.label_column
    if label_column is None:
        label_column = "behavioral_label" if "behavioral_label" in raw_data else "transition_label"
    data = add_audit_columns(raw_data, label_column=label_column)
    print(f"Auditing label column: {label_column}")
    if "transition_geometry_reliable" in data.columns:
        print(
            "Geometry-reliable transitions: "
            f"{data['transition_geometry_reliable'].mean():.1%}\n"
        )
    print_skip_distribution(data, label_column=label_column)
    print_line_return_audit(data, label_column=label_column)
    print_samples(data, label_column=label_column, samples=args.samples, seed=args.seed)


if __name__ == "__main__":
    main()
