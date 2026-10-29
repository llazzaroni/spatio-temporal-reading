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
LINGUISTIC_PREDICTORS = {
    "word_surprisal": {"raw": "word_level_surprisal", "missing": "word_level_surprisal_nan", "input_offset": 1, "label": "Word surprisal"},
    "char_surprisal": {"raw": "char_level_surp", "missing": "char_level_surp_nan", "input_offset": 0, "label": "Character surprisal"},
    # The Transformer receives -log(raw frequency), not raw frequency.
    "lexical_rarity": {"raw": "freq", "missing": "freq_nan", "input_offset": 3, "label": "Lexical rarity (-log frequency)"},
}


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


def prepare_empirical_controls(events: pd.DataFrame, raw: pd.DataFrame, training_stats: dict[str, dict[str, float]]) -> pd.DataFrame:
    raw = raw.sort_values(["text", "reader", "fixid"], kind="stable").copy()
    group = raw.groupby(["text", "reader"], sort=False)
    raw["previous_duration"] = group["dur"].shift()
    raw["previous_dx"] = group["dx"].shift()
    raw["previous_dy"] = group["dy"].shift()
    raw["previous_word_length"] = group["len"].shift()
    for name, spec in LINGUISTIC_PREDICTORS.items():
        raw[f"previous_{name}"] = group[spec["raw"]].shift()
        raw[f"previous_{name}_missing"] = group[spec["missing"]].shift()
    controls = raw[[
        "text", "reader", "fixid", "previous_duration", "previous_dx", "previous_dy",
        "previous_word_length",
        *[f"previous_{name}" for name in LINGUISTIC_PREDICTORS],
        *[f"previous_{name}_missing" for name in LINGUISTIC_PREDICTORS],
    ]]
    events = events.merge(controls, on=["text", "reader", "fixid"], how="left", validate="one_to_one")
    events["line_position"] = events["previous_word_position_in_line"] / events["previous_words_in_line"]
    events["regression_observed"] = events["behavioral_label"].eq("regression").astype(int)
    events["scanpath"] = events["text"].astype(str) + "::" + events["reader"].astype(str)
    events["previous_log_duration"] = np.log(events["previous_duration"].clip(lower=1e-6))
    events["previous_lexical_rarity"] = -np.log(events["previous_lexical_rarity"].clip(lower=1e-12))
    model_columns = [
        *[f"previous_{name}" for name in LINGUISTIC_PREDICTORS],
        "line_position", "previous_log_duration", "previous_dx", "previous_dy", "previous_word_length",
    ]
    events = events.loc[(events[[f"previous_{name}_missing" for name in LINGUISTIC_PREDICTORS]] == 0).all(axis=1)].copy()
    events = events.dropna(subset=model_columns)
    for column in model_columns:
        name = column.removeprefix("previous_")
        if name in training_stats:
            mean, sd = training_stats[name]["mean"], training_stats[name]["sd"]
        else:
            mean, sd = events[column].mean(), events[column].std(ddof=0)
        events[f"{column}_z"] = (events[column] - mean) / sd
    return events


def empirical_logistic(events: pd.DataFrame, weights: np.ndarray | None = None):
    predictors = [
        *[f"previous_{name}_z" for name in LINGUISTIC_PREDICTORS],
        "line_position_z", "previous_log_duration_z", "previous_dx_z", "previous_dy_z", "previous_word_length_z",
    ]
    import patsy
    formula = "regression_observed ~ " + " + ".join(predictors) + " + C(reader) + C(text)"
    y, x = patsy.dmatrices(formula, events, return_type="dataframe")
    kwargs = {} if weights is None else {"freq_weights": weights}
    fit = sm.GLM(y, x, family=sm.families.Binomial(), **kwargs).fit(
        cov_type="cluster" if weights is None else "nonrobust",
        cov_kwds={"groups": events["scanpath"]} if weights is None else None,
    )
    probabilities = np.asarray(fit.predict(x))
    beta = {name: float(fit.params[f"previous_{name}_z"]) for name in LINGUISTIC_PREDICTORS}
    ame = {name: value * float(np.mean(probabilities * (1 - probabilities))) for name, value in beta.items()}
    return fit, beta, ame


def bootstrap_empirical(events: pd.DataFrame, replicates: int, seed: int, progress: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    codes, names = pd.factorize(events["scanpath"], sort=True)
    draws: list[dict[str, float]] = []
    for _ in tqdm(range(replicates), desc="Empirical scanpath bootstrap", disable=not progress):
        counts = np.bincount(rng.integers(len(names), size=len(names)), minlength=len(names))
        weights = counts[codes]
        try:
            _, draw_beta, draw_ame = empirical_logistic(events, weights)
            draws.append({**{f"beta_{name}": value for name, value in draw_beta.items()}, **{f"ame_{name}": value for name, value in draw_ame.items()}})
        except (np.linalg.LinAlgError, ValueError):
            continue
    return pd.DataFrame(draws)


def save_figures(events: pd.DataFrame, beta: dict[str, float], empirical_bootstrap: pd.DataFrame, model_bootstrap: pd.DataFrame, output: Path) -> None:
    """Save the requested empirical-coefficient and model-sensitivity figures."""
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for axis, (name, spec) in zip(axes, LINGUISTIC_PREDICTORS.items()):
        values = empirical_bootstrap[f"beta_{name}"].dropna()
        axis.hist(values, bins=30, color="#54A24B")
        axis.axvline(beta[name], color="black", linestyle="--", label="Point estimate")
        axis.axvline(0, color="grey", linewidth=1)
        axis.set(title=spec["label"], xlabel="Log-odds per +1 training SD", ylabel="Bootstrap draws")
        axis.legend(fontsize=8)
    figure.suptitle("Empirical conditional associations with observed regressions")
    figure.savefig(output / "empirical_linguistic_coefficients.png", dpi=220)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for axis, (name, spec) in zip(axes, LINGUISTIC_PREDICTORS.items()):
        effects = events[f"model_effect_{name}_per_sd"].dropna()
        axis.hist(effects, bins=35, color="#B279A2")
        axis.axvline(0, color="black", linewidth=1)
        axis.axvline(effects.mean(), color="black", linestyle="--", label="Mean")
        axis.set(title=spec["label"], xlabel="Regression-probability change per +1 training SD", ylabel="Events")
        axis.legend(fontsize=8)
    figure.suptitle("Transformer partial sensitivity to linguistic predictors")
    figure.savefig(output / "transformer_linguistic_sensitivities.png", dpi=220)
    plt.close(figure)


def main(args: argparse.Namespace) -> None:
    if args.filtering != "raw":
        raise ValueError("This experiment is intentionally defined for --filtering raw.")
    checkpoint = args.checkpoint_path or args.data / "saccade" / "TRANSFORMER_RAW_COV_LM" / "best_model.pt"
    output = (args.output or checkpoint.parent / "regression_surprisal_analysis").resolve()
    output.mkdir(parents=True, exist_ok=True)
    model, uses_lm = checkpoint_model(checkpoint)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is not available.")
    model.to(device)
    dataset = (MecoDatasetLM if uses_lm else MecoDataset)(mode="test", filtering="raw", datadir=args.data)
    labels = pd.read_csv(args.labels)
    regions = GeometryRegions(pd.read_csv(args.texts), args.vertical_threshold, args.return_position_threshold)
    # Raw dataframe is also used for empirical previous-fixation covariates.
    raw_name = "hp_augmented_meco_100_1000_1_10_model_tokens.csv" if uses_lm else "hp_augmented_meco_100_1000_1_10_model.csv"
    raw = pd.read_csv(args.data / raw_name)
    train_dataset = (MecoDatasetLM if uses_lm else MecoDataset)(mode="train", filtering="raw", datadir=args.data)
    train_sessions = pd.DataFrame([item[:2] for item in train_dataset.items], columns=["text", "reader"])
    training_raw = raw.merge(train_sessions, on=["text", "reader"], how="inner")
    training_stats: dict[str, dict[str, float]] = {}
    for name, spec in LINGUISTIC_PREDICTORS.items():
        values = training_raw.loc[training_raw[spec["missing"]].eq(0), spec["raw"]].astype(float)
        if name == "lexical_rarity":
            values = -np.log(values.clip(lower=1e-12))
        training_stats[name] = {"mean": float(values.mean()), "sd": float(values.std(ddof=0))}
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
            inputs = inputs.to(device)
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
                event_count = len(indices)
                batch = prefix.unsqueeze(0).repeat(2 * len(LINGUISTIC_PREDICTORS) * event_count, 1, 1)
                feature_start = 4 + len(dataset.reader_to_idx)
                for predictor_index, (name, spec) in enumerate(LINGUISTIC_PREDICTORS.items()):
                    plus_rows = predictor_index * 2 * event_count + np.arange(event_count)
                    minus_rows = predictor_index * 2 * event_count + event_count + np.arange(event_count)
                    column = feature_start + spec["input_offset"]
                    batch[plus_rows, indices, column] += training_stats[name]["sd"]
                    batch[minus_rows, indices, column] -= training_stats[name]["sd"]
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
                    row["model_regression_probability"] = base_p
                    row["model_regression_numerator"] = base_num
                    row["model_primary_denominator"] = base_den
                    for predictor_index, name in enumerate(LINGUISTIC_PREDICTORS):
                        plus_row = predictor_index * 2 * event_count + local
                        minus_row = predictor_index * 2 * event_count + event_count + local
                        plus_p, _, _, _, _ = probability(plus_minus, plus_row, target)
                        minus_p, _, _, _, _ = probability(plus_minus, minus_row, target)
                        row[f"model_probability_{name}_plus_1sd"] = plus_p
                        row[f"model_probability_{name}_minus_1sd"] = minus_p
                        row[f"model_effect_{name}_per_sd"] = (plus_p - minus_p) / 2
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
    events = prepare_empirical_controls(events, raw, training_stats)
    fit, beta, ame = empirical_logistic(events)
    empirical_bootstrap = bootstrap_empirical(events, args.bootstrap_replicates, args.bootstrap_seed, args.progress)
    codes, names = pd.factorize(events["scanpath"], sort=True)
    rng = np.random.default_rng(args.bootstrap_seed)
    model_bootstrap: list[dict[str, float]] = []
    for _ in tqdm(range(args.bootstrap_replicates), desc="Transformer-effect bootstrap", disable=not args.progress):
        counts = np.bincount(rng.integers(len(names), size=len(names)), minlength=len(names))
        model_bootstrap.append({name: float(np.average(events[f"model_effect_{name}_per_sd"], weights=counts[codes])) for name in LINGUISTIC_PREDICTORS})
    model_bootstrap_df = pd.DataFrame(model_bootstrap)
    empirical_summary = {name: {
        "beta_log_odds_per_1sd": beta[name], "odds_ratio_per_1sd": float(np.exp(beta[name])),
        "average_marginal_effect_per_1sd": ame[name],
        "cluster_robust_ci_beta": fit.conf_int().loc[f"previous_{name}_z"].astype(float).tolist(),
        "scanpath_bootstrap_ci_beta": np.quantile(empirical_bootstrap[f"beta_{name}"], [.025, .975]).tolist(),
    } for name in LINGUISTIC_PREDICTORS}
    model_summary = {name: {
        "mean_effect_per_1sd": float(events[f"model_effect_{name}_per_sd"].mean()),
        "median_effect_per_1sd": float(events[f"model_effect_{name}_per_sd"].median()),
        "scanpath_bootstrap_ci_mean_effect": np.quantile(model_bootstrap_df[name], [.025, .975]).tolist(),
    } for name in LINGUISTIC_PREDICTORS} | {
        "mean_conditional_regression_probability": float(events["model_regression_probability"].mean()),
        "observed_regression_frequency": float(events["regression_observed"].mean()),
    }
    events.to_csv(output / "test_regression_surprisal_events.csv", index=False)
    fit.summary2().tables[1].to_csv(output / "empirical_logistic_coefficients.csv")
    empirical_bootstrap.to_csv(output / "empirical_logistic_scanpath_bootstrap.csv", index=False)
    model_bootstrap_df.to_csv(output / "model_effect_scanpath_bootstrap.csv", index=False)
    summary_table = pd.DataFrame([{
        "predictor": name,
        "empirical_coefficient": beta[name],
        "empirical_ci_lower": empirical_summary[name]["scanpath_bootstrap_ci_beta"][0],
        "empirical_ci_upper": empirical_summary[name]["scanpath_bootstrap_ci_beta"][1],
        "mean_transformer_sensitivity": model_summary[name]["mean_effect_per_1sd"],
        "transformer_ci_lower": model_summary[name]["scanpath_bootstrap_ci_mean_effect"][0],
        "transformer_ci_upper": model_summary[name]["scanpath_bootstrap_ci_mean_effect"][1],
    } for name in LINGUISTIC_PREDICTORS])
    summary_table.to_csv(output / "linguistic_predictor_summary.csv", index=False)
    save_figures(events, beta, empirical_bootstrap, model_bootstrap_df, output)
    summary = {
        "checkpoint": str(Path(checkpoint).resolve()), "uses_lm_embeddings": uses_lm,
        "device": str(device),
        "included_events": int(len(events)), "training_feature_statistics": training_stats,
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
    parser.add_argument(
        "--perturbation-batch-size", type=int, default=8,
        help="Number of events per GPU batch; each event expands to six +/- predictor variants (default: 8).",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--bootstrap-seed", type=int, default=20261029)
    parser.add_argument("--max-events", type=int, help="Debug option: stop after this many eligible events.")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True, help="Show tqdm progress bars (default: true).")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Inference device; auto selects CUDA when available.")
    main(parser.parse_args())
