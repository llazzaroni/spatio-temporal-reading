"""Joint scalar-linguistic difficulty sensitivity of predicted behaviours.

For each eligible raw held-out transition, perturb only the preceding
fixation's character surprisal, word surprisal, and lexical-rarity inputs by
their respective training-set standard deviations, jointly.  Integrate the
COV mixture over geometry-defined primary behavioural regions.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-joint-linguistic")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from attribution.behavioral_responsibilities import checkpoint_model
from attribution.regression_surprisal_analysis import (
    GeometryRegions, LINGUISTIC_PREDICTORS, PRIMARY, session_labels,
    rectangle_mass,
)
from spatio_temporal_reading.data.data import MecoDataset
from spatio_temporal_reading.data.dataLM import MecoDatasetLM


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
LABELS = DATA_DIR / "hp_augmented_meco_100_1000_1_10_model_filtered_behavioral_labels.csv"
TEXTS = DATA_DIR / "hp_eng_texts_100_1000_1_10.csv"


def behavior_probabilities(payload):
    """Worker: calculate all conditional primary-behaviour probabilities."""
    regions, previous_x, previous_y, parameter_sets = payload
    translate = np.array([previous_x, previous_x, previous_y, previous_y])
    results = []
    for weights, means, covariances in parameter_sets:
        component_masses = {
            behavior: rectangle_mass(rectangles - translate, means, covariances)
            for behavior, rectangles in regions.items()
        }
        denominator_mass = sum(component_masses.values())
        denominator = float(weights @ denominator_mass)
        probabilities = {
            behavior: float(weights @ mass) / denominator if denominator > 1e-12 else np.nan
            for behavior, mass in component_masses.items()
        }
        results.append(probabilities)
    return results


def training_statistics(dataset_type, data: Path, uses_lm: bool):
    train = dataset_type(mode="train", filtering="raw", datadir=data)
    raw_name = "hp_augmented_meco_100_1000_1_10_model_tokens.csv" if uses_lm else "hp_augmented_meco_100_1000_1_10_model.csv"
    raw = pd.read_csv(data / raw_name)
    sessions = pd.DataFrame([item[:2] for item in train.items], columns=["text", "reader"])
    raw = raw.merge(sessions, on=["text", "reader"], how="inner")
    valid = (raw[[spec["missing"] for spec in LINGUISTIC_PREDICTORS.values()]] == 0).all(axis=1)
    values = pd.DataFrame({name: raw.loc[valid, spec["raw"]].astype(float) for name, spec in LINGUISTIC_PREDICTORS.items()})
    values["lexical_rarity"] = -np.log(values["lexical_rarity"].clip(lower=1e-12))
    means, sds = values.mean(), values.std(ddof=0)
    z = (values - means) / sds
    return {name: {"mean": float(means[name]), "sd": float(sds[name])} for name in LINGUISTIC_PREDICTORS}, z.corr(), np.linalg.inv(z.cov().to_numpy())


def main(args):
    checkpoint = args.checkpoint_path or args.data / "saccade" / "TRANSFORMER_RAW_COV_LM" / "best_model.pt"
    output = (args.output or checkpoint.parent / "joint_linguistic_behavior_sensitivity").resolve()
    output.mkdir(parents=True, exist_ok=True)
    model, uses_lm = checkpoint_model(checkpoint)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    model.to(device).eval()
    dataset_type = MecoDatasetLM if uses_lm else MecoDataset
    dataset = dataset_type(mode="test", filtering="raw", datadir=args.data)
    stats, correlations, inverse_covariance = training_statistics(dataset_type, args.data, uses_lm)
    labels = pd.read_csv(args.labels)
    regions = GeometryRegions(pd.read_csv(args.texts), args.vertical_threshold, args.return_position_threshold)
    feature_start = 4 + len(dataset.reader_to_idx)
    offsets = [spec["input_offset"] for spec in LINGUISTIC_PREDICTORS.values()]
    mask_offsets = [4, 5, 7]
    executor = ProcessPoolExecutor(max_workers=args.integration_workers, mp_context=multiprocessing.get_context("spawn")) if args.integration_workers > 1 else None
    events, processed = [], 0
    try:
        with torch.no_grad():
            for item_index, item in enumerate(tqdm(dataset.items, desc="Held-out scanpaths", disable=not args.progress)):
                text, reader = item[:2]
                inputs, _, _ = dataset[item_index]
                inputs = inputs.to(device)
                metadata = dataset.meco_df[(dataset.meco_df.text == text) & (dataset.meco_df.reader == reader)].copy()
                frame = session_labels(labels, metadata)
                valid = frame.behavioral_label.isin(PRIMARY) & frame.is_consecutive_original_fixation.eq(True)
                valid &= frame.index.to_series().gt(0).to_numpy()
                valid &= inputs[:, [feature_start + offset for offset in mask_offsets]].eq(0).all(dim=1).cpu().numpy()
                indices = np.flatnonzero(valid.to_numpy())
                if args.max_events is not None:
                    remaining = args.max_events - processed
                    if remaining <= 0:
                        break
                    indices = indices[:remaining]
                if not len(indices):
                    continue
                baseline = model(inputs.unsqueeze(0))
                for start in range(0, len(indices), args.perturbation_batch_size):
                    batch_indices = indices[start:start + args.perturbation_batch_size]
                    prefix = inputs[:int(batch_indices.max()) + 1]
                    count = len(batch_indices)
                    variants = prefix.unsqueeze(0).repeat(2 * count, 1, 1)
                    for value_index, (name, spec) in enumerate(LINGUISTIC_PREDICTORS.items()):
                        column = feature_start + spec["input_offset"]
                        variants[np.arange(count), batch_indices, column] += stats[name]["sd"]
                        variants[count + np.arange(count), batch_indices, column] -= stats[name]["sd"]
                    plus_minus = model(variants)
                    base_selected = tuple(t[0, batch_indices].cpu().numpy() for t in (baseline[0], baseline[1], baseline[3]))
                    plus_selected = tuple(t[np.arange(count), batch_indices].cpu().numpy() for t in (plus_minus[0], plus_minus[1], plus_minus[3]))
                    minus_selected = tuple(t[count + np.arange(count), batch_indices].cpu().numpy() for t in (plus_minus[0], plus_minus[1], plus_minus[3]))
                    rows_payloads = []
                    for local, target in enumerate(batch_indices):
                        row = frame.iloc[target].copy()
                        category_regions = regions.rectangles(row)
                        parameter_sets = [tuple(x[local] for x in selected) for selected in (base_selected, plus_selected, minus_selected)]
                        rows_payloads.append((row, (category_regions, float(row.previous_x), float(row.previous_y), parameter_sets), target))
                    payloads = [payload for _, payload, _ in rows_payloads]
                    probability_results = list(executor.map(behavior_probabilities, payloads)) if executor else [behavior_probabilities(payload) for payload in payloads]
                    for (row, _, target), probabilities in zip(rows_payloads, probability_results):
                        original, plus, minus = probabilities
                        for behavior in PRIMARY:
                            row[f"probability_{behavior}"] = original[behavior]
                            row[f"probability_{behavior}_high"] = plus[behavior]
                            row[f"probability_{behavior}_low"] = minus[behavior]
                            row[f"sensitivity_{behavior}"] = (plus[behavior] - minus[behavior]) / 2
                        prior_values = inputs[target, [feature_start + offset for offset in offsets]].cpu().numpy()
                        for value, name in zip(prior_values, LINGUISTIC_PREDICTORS):
                            row[f"prior_z_{name}"] = (value - stats[name]["mean"]) / stats[name]["sd"]
                        events.append(pd.DataFrame([row]))
                        processed += 1
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    events = pd.concat(events, ignore_index=True)
    events["scanpath"] = events.text.astype(str) + "::" + events.reader.astype(str)
    sensitivity_sum = events[[f"sensitivity_{behavior}" for behavior in PRIMARY]].sum(axis=1)
    z = events[[f"prior_z_{name}" for name in LINGUISTIC_PREDICTORS]].to_numpy()
    mahalanobis_high = np.einsum("ij,jk,ik->i", z + 1, inverse_covariance, z + 1)
    mahalanobis_low = np.einsum("ij,jk,ik->i", z - 1, inverse_covariance, z - 1)
    codes, names = pd.factorize(events.scanpath, sort=True)
    rng = np.random.default_rng(args.seed)
    bootstrap = []
    for _ in tqdm(range(args.bootstrap_replicates), desc="Scanpath bootstrap", disable=not args.progress):
        counts = np.bincount(rng.integers(len(names), size=len(names)), minlength=len(names))
        bootstrap.append({behavior: float(np.average(events[f"sensitivity_{behavior}"], weights=counts[codes])) for behavior in PRIMARY})
    bootstrap = pd.DataFrame(bootstrap)
    summary_rows = []
    for behavior in PRIMARY:
        summary_rows.append({"behavior": behavior, "mean_sensitivity": events[f"sensitivity_{behavior}"].mean(), "ci_lower": bootstrap[behavior].quantile(.025), "ci_upper": bootstrap[behavior].quantile(.975)})
    summary_table = pd.DataFrame(summary_rows)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for axis, behavior in zip(axes, ("regression", "refixation", "skip")):
        values = events[f"sensitivity_{behavior}"]
        axis.hist(values, bins=35, color="#4C78A8")
        axis.axvline(0, color="black", linewidth=1)
        axis.axvline(values.mean(), color="black", linestyle="--", label="Mean")
        axis.set(title=behavior.replace("_", " "), xlabel="Probability change per joint +1 SD step", ylabel="Events")
        axis.legend(fontsize=8)
    figure.savefig(output / "joint_linguistic_behavior_sensitivities.png", dpi=220)
    plt.close(figure)
    correlations.to_csv(output / "training_linguistic_correlations.csv")
    events.to_csv(output / "joint_linguistic_behavior_events.csv", index=False)
    summary_table.to_csv(output / "joint_linguistic_behavior_summary.csv", index=False)
    bootstrap.to_csv(output / "joint_linguistic_behavior_bootstrap.csv", index=False)
    summary = {"checkpoint": str(Path(checkpoint).resolve()), "device": str(device), "included_events": len(events), "training_feature_statistics": stats, "mean_abs_probability_conservation_error": float(sensitivity_sum.abs().mean()), "mean_mahalanobis_distance_high": float(mahalanobis_high.mean()), "mean_mahalanobis_distance_low": float(mahalanobis_low.mean())}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--texts", type=Path, default=TEXTS)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--vertical-threshold", type=float, default=.5)
    parser.add_argument("--return-position-threshold", type=float, default=.5)
    parser.add_argument("--perturbation-batch-size", type=int, default=16)
    parser.add_argument("--integration-workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", min(8, os.cpu_count() or 1))))
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-events", type=int)
    main(parser.parse_args())
