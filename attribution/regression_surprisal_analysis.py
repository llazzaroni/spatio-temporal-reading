"""Test whether previous-word surprisal predicts regression behaviour.

The analysis uses raw COV saccade Transformers, but retains only transitions
whose two word-aligned fixations are consecutive in the original scanpath.
For each event it integrates every mixture component's *bivariate* spatial
Gaussian over text-layout regions that the geometry-aware label rules would
classify as a regression or as any primary behaviour.  The reported model
probability is therefore

    P(regression | geometrically classifiable landing position, H).

It also fits a clustered empirical logistic regression and estimates a local
Transformer effect by replacing the previous word's surprisal by +/- one
standard deviation while holding every other input fixed.

Example::

    python -m attribution.regression_surprisal_analysis \
      --filtering raw \
      --checkpoint-path data/saccade/TRANSFORMER_RAW_COV_LM/best_model.pt \
      --output data/saccade/TRANSFORMER_RAW_COV_LM/regression_surprisal_analysis
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# The execution environment can have a non-writable home-directory cache.
# Set this before importing pyplot so plots do not spend time rebuilding it.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-regression-surprisal")
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from scipy.stats import multivariate_normal
from tqdm.auto import tqdm

from attribution.behavioral_responsibilities import checkpoint_model
from data_preprocessing.label_filtered_fixations_geometry import text_layout
from spatio_temporal_reading.data.data import MecoDataset
from spatio_temporal_reading.data.dataLM import MecoDatasetLM


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_LABELS = DATA_DIR / "hp_augmented_meco_100_1000_1_10_model_filtered_behavioral_labels.csv"
TEXTS = DATA_DIR / "hp_eng_texts_100_1000_1_10.csv"
PRIMARY = ("forward", "skip", "refixation", "regression", "line_return")


def _normalise_text(value: object) -> int:
    return int(float(value))


class GeometryRegions:
    """Construct prospective primary/regression landing rectangles from layout."""

    def __init__(self, texts: pd.DataFrame, vertical_threshold: float = .5, return_threshold: float = .5):
        ranks, centres, spacing = text_layout(texts)
        self.ranks = ranks.reset_index()
        self.centres = centres
        self.spacing = spacing
        self.vertical_threshold = vertical_threshold
        self.return_threshold = return_threshold
        words = texts.loc[texts["ianum_word"].notna(), [
            "text_id", "line", "ianum_word", "word_bbox_x1", "word_bbox_x2",
            "word_bbox_y1", "word_bbox_y2",
        ]].dropna(subset=["word_bbox_x1", "word_bbox_x2", "word_bbox_y1", "word_bbox_y2"])
        # Word-level boxes are repeated over characters.  Aggregate after
        # removing character rows without a complete word box so one missing
        # character annotation cannot discard the entire candidate word.
        words = words.groupby(["text_id", "line", "ianum_word"], as_index=False).first()
        words = words.merge(self.ranks, on=["text_id", "line", "ianum_word"], validate="one_to_one")
        words["line_center_y"] = words.set_index(["text_id", "line"]).index.map(self.centres)
        self.words = {int(text): group.reset_index(drop=True) for text, group in words.groupby("text_id")}

    def rectangles(self, row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Return non-overlapping absolute-coordinate rectangles for R and E.

        The reliability conditions used here are exactly those decidable from
        the previous fixation, a proposed landing coordinate, and the fixed
        text geometry.  The three-fixation audit flag is intentionally absent.
        """
        if not bool(row["previous_word_assignment_reliable"]):
            return np.empty((0, 4)), np.empty((0, 4))
        text = _normalise_text(row["text"])
        previous_line = row["previous_line"]
        previous_word = row["previous_ianum_word"]
        previous_x = float(row["previous_x"])
        previous_y = float(row["previous_y"])
        previous_center = float(row["previous_line_center_y"])
        previous_relative = float(row["previous_word_position_in_line"] / row["previous_words_in_line"])
        line_spacing = float(row["line_spacing"])
        if not np.isfinite(line_spacing) or line_spacing <= 0:
            return np.empty((0, 4)), np.empty((0, 4))

        regression: list[list[float]] = []
        primary: list[list[float]] = []
        for candidate in self.words[text].itertuples(index=False):
            if not np.isfinite([candidate.word_bbox_x1, candidate.word_bbox_x2, candidate.word_bbox_y1, candidate.word_bbox_y2, candidate.line_center_y]).all():
                continue
            line_difference = candidate.line - previous_line
            word_difference = candidate.ianum_word - previous_word
            # Restrict y to the part of the word box that satisfies the same
            # vertical-consistency check as the observed-label procedure.
            expected_y = candidate.line_center_y - previous_center
            low_y = previous_y + expected_y - self.vertical_threshold * line_spacing
            high_y = previous_y + expected_y + self.vertical_threshold * line_spacing
            y1, y2 = max(candidate.word_bbox_y1, low_y), min(candidate.word_bbox_y2, high_y)
            if y1 >= y2:
                continue

            label: str | None = None
            x1, x2 = candidate.word_bbox_x1, candidate.word_bbox_x2
            if line_difference < 0:
                label = "regression"
            elif line_difference == 0 and word_difference == 0:
                label = "refixation"
            elif line_difference == 0 and word_difference == 1:
                x1 = max(x1, previous_x)
                label = "forward"
            elif line_difference == 0 and word_difference > 1:
                x1 = max(x1, previous_x)
                label = "skip"
            elif line_difference == 0 and word_difference < 0:
                x2 = min(x2, previous_x)
                label = "regression"
            elif (
                line_difference == 1
                and previous_relative >= self.return_threshold
                and candidate.word_position_in_line / candidate.words_in_line <= self.return_threshold
            ):
                x2 = min(x2, previous_x)
                label = "line_return"
            if label is None or x1 >= x2:
                continue
            rectangle = [float(x1), float(x2), float(y1), float(y2)]
            primary.append(rectangle)
            if label == "regression":
                regression.append(rectangle)
        return np.asarray(regression, dtype=float).reshape(-1, 4), np.asarray(primary, dtype=float).reshape(-1, 4)


def rectangle_mass(rectangles: np.ndarray, means: np.ndarray, covariances: np.ndarray) -> np.ndarray:
    """Mass of disjoint axis-aligned rectangles under each 2-D Gaussian."""
    components = len(means)
    masses = np.zeros(components)
    if not len(rectangles):
        return masses
    for component, (mean, covariance) in enumerate(zip(means, covariances)):
        standard_deviation = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
        # A rectangle outside eight marginal SDs has negligible mass; dropping
        # it makes the exact CDF calculation practical without changing output
        # at meaningful precision.
        lower = mean - 8 * standard_deviation
        upper = mean + 8 * standard_deviation
        rect = rectangles[
            (rectangles[:, 1] >= lower[0]) & (rectangles[:, 0] <= upper[0])
            & (rectangles[:, 3] >= lower[1]) & (rectangles[:, 2] <= upper[1])
        ]
        if not len(rect):
            continue
        # The model distribution is over displacement, while layout boxes are
        # absolute coordinates; callers have already translated the rectangles.
        points = np.concatenate([
            rect[:, [1, 3]], rect[:, [0, 3]], rect[:, [1, 2]], rect[:, [0, 2]],
        ])
        cdf = multivariate_normal.cdf(points, mean=mean, cov=covariance)
        masses[component] = np.sum(cdf[:len(rect)] - cdf[len(rect):2 * len(rect)] - cdf[2 * len(rect):3 * len(rect)] + cdf[3 * len(rect):])
    return np.clip(masses, 0.0, 1.0)


def conditional_regression_probability(
    weights: np.ndarray, means: np.ndarray, covariances: np.ndarray,
    regression_rectangles: np.ndarray, primary_rectangles: np.ndarray,
    previous_x: float, previous_y: float,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Return conditional regression probability and numerator/denominator."""
    translate = np.array([previous_x, previous_x, previous_y, previous_y])
    reg_mass = rectangle_mass(regression_rectangles - translate, means, covariances)
    primary_mass = rectangle_mass(primary_rectangles - translate, means, covariances)
    numerator = float(weights @ reg_mass)
    denominator = float(weights @ primary_mass)
    return (numerator / denominator if denominator > 1e-12 else np.nan), numerator, denominator, reg_mass, primary_mass


def session_labels(labels: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "text", "reader", "fixid", "behavioral_label", "is_consecutive_original_fixation",
        "previous_word_assignment_reliable", "previous_line", "previous_ianum_word",
        "previous_x", "previous_y", "previous_line_center_y", "previous_word_position_in_line",
        "previous_words_in_line", "line_spacing", "word_position_in_line", "words_in_line",
    ]
    result = metadata[["text", "reader", "fixid"]].merge(
        labels[keep], how="left", on=["text", "reader", "fixid"], validate="one_to_one"
    )
    return result


def prepare_empirical_controls(events: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.sort_values(["text", "reader", "fixid"], kind="stable").copy()
    group = raw.groupby(["text", "reader"], sort=False)
    raw["previous_duration"] = group["dur"].shift()
    raw["previous_dx"] = group["dx"].shift()
    raw["previous_dy"] = group["dy"].shift()
    raw["previous_word_length"] = group["len"].shift()
    raw["previous_frequency"] = group["freq"].shift()
    raw["previous_word_surprisal"] = group["word_level_surprisal"].shift()
    raw["previous_word_surprisal_missing"] = group["word_level_surprisal_nan"].shift()
    controls = raw[[
        "text", "reader", "fixid", "previous_duration", "previous_dx", "previous_dy",
        "previous_word_length", "previous_frequency", "previous_word_surprisal",
        "previous_word_surprisal_missing",
    ]]
    events = events.merge(controls, on=["text", "reader", "fixid"], how="left", validate="one_to_one")
    events["line_position"] = events["previous_word_position_in_line"] / events["previous_words_in_line"]
    events["regression_observed"] = events["behavioral_label"].eq("regression").astype(int)
    events["scanpath"] = events["text"].astype(str) + "::" + events["reader"].astype(str)
    events["previous_log_duration"] = np.log(events["previous_duration"].clip(lower=1e-6))
    events["previous_frequency"] = -np.log(events["previous_frequency"].clip(lower=1e-12))
    model_columns = [
        "previous_word_surprisal", "line_position", "previous_log_duration", "previous_dx",
        "previous_dy", "previous_word_length", "previous_frequency",
    ]
    events = events.loc[events["previous_word_surprisal_missing"].eq(0)].copy()
    events = events.dropna(subset=model_columns)
    for column in model_columns:
        mean, sd = events[column].mean(), events[column].std(ddof=0)
        events[f"{column}_z"] = (events[column] - mean) / sd
    return events


def empirical_logistic(events: pd.DataFrame, weights: np.ndarray | None = None):
    predictors = [
        "previous_word_surprisal_z", "line_position_z", "previous_log_duration_z", "previous_dx_z",
        "previous_dy_z", "previous_word_length_z", "previous_frequency_z",
    ]
    import patsy
    formula = "regression_observed ~ " + " + ".join(predictors) + " + C(reader) + C(text)"
    y, x = patsy.dmatrices(formula, events, return_type="dataframe")
    kwargs = {} if weights is None else {"freq_weights": weights}
    fit = sm.GLM(y, x, family=sm.families.Binomial(), **kwargs).fit(
        cov_type="cluster" if weights is None else "nonrobust",
        cov_kwds={"groups": events["scanpath"]} if weights is None else None,
    )
    beta = float(fit.params["previous_word_surprisal_z"])
    probabilities = np.asarray(fit.predict(x))
    ame = beta * float(np.mean(probabilities * (1 - probabilities)))
    return fit, beta, ame


def bootstrap_empirical(events: pd.DataFrame, replicates: int, seed: int, progress: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    codes, names = pd.factorize(events["scanpath"], sort=True)
    beta, ame = [], []
    for _ in tqdm(range(replicates), desc="Empirical scanpath bootstrap", disable=not progress):
        counts = np.bincount(rng.integers(len(names), size=len(names)), minlength=len(names))
        weights = counts[codes]
        try:
            _, draw_beta, draw_ame = empirical_logistic(events, weights)
            beta.append(draw_beta)
            ame.append(draw_ame)
        except (np.linalg.LinAlgError, ValueError):
            continue
    return pd.DataFrame({"beta": beta, "odds_ratio": np.exp(beta), "average_marginal_effect": ame})


def save_figure(events: pd.DataFrame, empirical: dict[str, float], bootstrap: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.ravel()
    valid = events.dropna(subset=["model_regression_probability"])
    axes[0].hist(valid["model_regression_probability"], bins=35, color="#4C78A8")
    axes[0].set(title="Predicted conditional regression probability", xlabel="Probability", ylabel="Events")
    bins = pd.qcut(valid["model_regression_probability"], q=min(10, valid["model_regression_probability"].nunique()), duplicates="drop")
    calibration = valid.groupby(bins, observed=True).agg(predicted=("model_regression_probability", "mean"), observed=("regression_observed", "mean"))
    axes[1].plot(calibration["predicted"], calibration["observed"], "o-", color="#F58518")
    axes[1].plot([0, 1], [0, 1], "--", color="black")
    axes[1].set(title="Calibration on classifiable transitions", xlabel="Mean predicted probability", ylabel="Observed regression frequency", xlim=(0, 1), ylim=(0, 1))
    axes[2].hist(bootstrap["beta"], bins=30, color="#54A24B")
    axes[2].axvline(empirical["beta"], color="black", linestyle="--", label="Point estimate")
    axes[2].axvline(0, color="grey", linewidth=1)
    axes[2].set(title="Empirical surprisal coefficient", xlabel="Log-odds per +1 SD", ylabel="Bootstrap draws")
    axes[2].legend()
    effects = valid["model_surprisal_effect_per_sd"].dropna()
    axes[3].hist(effects, bins=35, color="#B279A2")
    axes[3].axvline(0, color="black", linewidth=1)
    axes[3].axvline(effects.mean(), color="black", linestyle="--", label="Mean")
    axes[3].set(title="Transformer local surprisal sensitivity", xlabel="Conditional regression-probability change per +1 SD", ylabel="Events")
    axes[3].legend()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def main(args: argparse.Namespace) -> None:
    if args.filtering != "raw":
        raise ValueError("This experiment is intentionally defined for --filtering raw.")
    checkpoint = args.checkpoint_path or args.data / "saccade" / "TRANSFORMER_RAW_COV_LM" / "best_model.pt"
    output = (args.output or checkpoint.parent / "regression_surprisal_analysis").resolve()
    output.mkdir(parents=True, exist_ok=True)
    model, uses_lm = checkpoint_model(checkpoint)
    dataset = (MecoDatasetLM if uses_lm else MecoDataset)(mode="test", filtering="raw", datadir=args.data)
    labels = pd.read_csv(args.labels)
    regions = GeometryRegions(pd.read_csv(args.texts), args.vertical_threshold, args.return_position_threshold)
    # Raw dataframe is also used for empirical previous-fixation covariates.
    raw = pd.read_csv(args.data / "hp_augmented_meco_100_1000_1_10_model_tokens.csv")
    surprisal_sd = raw.loc[raw["word_level_surprisal_nan"].eq(0), "word_level_surprisal"].std(ddof=0)
    event_frames: list[pd.DataFrame] = []
    processed_targets = 0
    # Dataset items carry an additional split/metadata field in some dataset
    # variants; only their first two entries identify the scanpath.
    test_sessions = pd.DataFrame([item[:2] for item in dataset.items], columns=["text", "reader"])
    candidate_labels = labels.merge(test_sessions, on=["text", "reader"], how="inner")
    eligible_total = int((
        candidate_labels["behavioral_label"].isin(PRIMARY)
        & candidate_labels["is_consecutive_original_fixation"].eq(True)
    ).sum())
    if args.max_events is not None:
        eligible_total = min(eligible_total, args.max_events)
    event_progress = tqdm(
        total=eligible_total,
        desc="Model perturbation events",
        unit="event",
        disable=not args.progress,
    )

    with torch.no_grad():
        for item_index, item in enumerate(tqdm(dataset.items, desc="Held-out scanpaths", unit="scanpath", disable=not args.progress)):
            text, reader = item[:2]
            inputs, _, _ = dataset[item_index]
            metadata = dataset.meco_df[(dataset.meco_df.text == text) & (dataset.meco_df.reader == reader)].copy()
            frame = session_labels(labels, metadata)
            valid = frame.behavioral_label.isin(PRIMARY) & frame.is_consecutive_original_fixation.astype(bool)
            # A target at index t is predicted from input[t], which contains
            # fixation t-1's covariates because the dataset shifts inputs.
            valid &= frame.index.to_series().gt(0).to_numpy()
            if not valid.any():
                continue
            baseline = model(inputs.unsqueeze(0))
            target_indices = np.flatnonzero(valid.to_numpy())
            if args.max_events is not None:
                remaining = args.max_events - processed_targets
                if remaining <= 0:
                    break
                target_indices = target_indices[:remaining]
            for start in range(0, len(target_indices), args.perturbation_batch_size):
                indices = target_indices[start:start + args.perturbation_batch_size]
                # Causality means that the output at target t is unchanged if
                # we discard tokens after t.  Prefix batching avoids carrying
                # a long raw scanpath through attention for every perturbation.
                prefix = inputs[:int(indices.max()) + 1]
                batch = prefix.unsqueeze(0).repeat(2 * len(indices), 1, 1)
                surprise_column = 5 + len(dataset.reader_to_idx)
                batch[np.arange(len(indices)), indices, surprise_column] += surprisal_sd
                batch[len(indices) + np.arange(len(indices)), indices, surprise_column] -= surprisal_sd
                plus_minus = model(batch)
                for local, target in enumerate(indices):
                    row = frame.iloc[target].copy()
                    regression_rectangles, primary_rectangles = regions.rectangles(row)
                    if not len(primary_rectangles):
                        continue
                    def probability(outputs, batch_index: int, event_index: int):
                        weights, means, _, covariances, _ = outputs
                        return conditional_regression_probability(
                            weights[batch_index, event_index].cpu().numpy(),
                            means[batch_index, event_index].cpu().numpy(),
                            covariances[batch_index, event_index].cpu().numpy(),
                            regression_rectangles, primary_rectangles, row.previous_x, row.previous_y,
                        )
                    base_p, base_num, base_den, base_reg_mass, base_primary_mass = probability(baseline, 0, target)
                    plus_p, _, _, _, _ = probability(plus_minus, local, target)
                    minus_p, _, _, _, _ = probability(plus_minus, len(indices) + local, target)
                    row["model_regression_probability"] = base_p
                    row["model_regression_numerator"] = base_num
                    row["model_primary_denominator"] = base_den
                    row["model_probability_plus_1sd"] = plus_p
                    row["model_probability_minus_1sd"] = minus_p
                    row["model_surprisal_effect_per_sd"] = (plus_p - minus_p) / 2
                    base_weights = baseline[0][0, target].cpu().numpy()
                    for component in range(len(base_weights)):
                        row[f"prior_weight_{component}"] = base_weights[component]
                        row[f"regression_region_mass_{component}"] = base_reg_mass[component]
                        row[f"primary_region_mass_{component}"] = base_primary_mass[component]
                    event_frames.append(pd.DataFrame([row]))
                    processed_targets += 1
                    event_progress.update(1)
    event_progress.close()

    events = pd.concat(event_frames, ignore_index=True)
    events = prepare_empirical_controls(events, raw)
    fit, beta, ame = empirical_logistic(events)
    empirical_bootstrap = bootstrap_empirical(events, args.bootstrap_replicates, args.bootstrap_seed, args.progress)
    model_effect = events["model_surprisal_effect_per_sd"].dropna()
    codes, names = pd.factorize(events["scanpath"], sort=True)
    rng = np.random.default_rng(args.bootstrap_seed)
    model_bootstrap = []
    for _ in tqdm(range(args.bootstrap_replicates), desc="Transformer-effect bootstrap", disable=not args.progress):
        counts = np.bincount(rng.integers(len(names), size=len(names)), minlength=len(names))
        model_bootstrap.append(np.average(events["model_surprisal_effect_per_sd"], weights=counts[codes]))
    empirical_summary = {
        "beta_log_odds_per_1sd": beta,
        "odds_ratio_per_1sd": float(np.exp(beta)),
        "average_marginal_effect_per_1sd": ame,
        "cluster_robust_ci_beta": [float(fit.conf_int().loc["previous_word_surprisal_z", 0]), float(fit.conf_int().loc["previous_word_surprisal_z", 1])],
        "scanpath_bootstrap_ci_beta": np.quantile(empirical_bootstrap["beta"], [.025, .975]).tolist(),
        "scanpath_bootstrap_ci_ame": np.quantile(empirical_bootstrap["average_marginal_effect"], [.025, .975]).tolist(),
    }
    model_summary = {
        "mean_effect_per_1sd": float(model_effect.mean()),
        "median_effect_per_1sd": float(model_effect.median()),
        "scanpath_bootstrap_ci_mean_effect": np.quantile(model_bootstrap, [.025, .975]).tolist(),
        "mean_conditional_regression_probability": float(events["model_regression_probability"].mean()),
        "observed_regression_frequency": float(events["regression_observed"].mean()),
    }
    events.to_csv(output / "test_regression_surprisal_events.csv", index=False)
    fit.summary2().tables[1].to_csv(output / "empirical_logistic_coefficients.csv")
    empirical_bootstrap.to_csv(output / "empirical_logistic_scanpath_bootstrap.csv", index=False)
    pd.DataFrame({"mean_model_effect_per_1sd": model_bootstrap}).to_csv(output / "model_effect_scanpath_bootstrap.csv", index=False)
    save_figure(events, {"beta": beta}, empirical_bootstrap, output / "regression_surprisal_summary.png")
    summary = {
        "checkpoint": str(Path(checkpoint).resolve()), "uses_lm_embeddings": uses_lm,
        "included_events": int(len(events)), "surprisal_sd_raw_units": float(surprisal_sd),
        "integration": "exact bivariate-normal rectangle CDF; regions are geometrically classifiable landing positions",
        "empirical": empirical_summary, "transformer": model_summary,
        "bootstrap_replicates": args.bootstrap_replicates,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote results to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--texts", type=Path, default=TEXTS)
    parser.add_argument("--filtering", choices=["raw"], default="raw")
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--vertical-threshold", type=float, default=.5)
    parser.add_argument("--return-position-threshold", type=float, default=.5)
    parser.add_argument("--perturbation-batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--bootstrap-seed", type=int, default=20261029)
    parser.add_argument("--max-events", type=int, help="Debug option: stop after this many eligible events.")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True, help="Show tqdm progress bars (default: true).")
    main(parser.parse_args())
