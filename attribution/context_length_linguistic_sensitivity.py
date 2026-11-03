"""Test whether joint linguistic sensitivity changes with scanpath context.

For every included held-out transition, the same preceding linguistic triplet
is increased or decreased by one training-set standard deviation.  The model
is evaluated with ``L`` preceding fixations only, for each requested context
length.  Thus only the accessible scanpath history changes across conditions.

Example::

    python -m attribution.context_length_linguistic_sensitivity \\
      --checkpoint-path data/saccade/TRANSFORMER_RAW_COV_LM/best_model.pt \\
      --output data/saccade/TRANSFORMER_RAW_COV_LM/context_length_linguistic_sensitivity \\
      --context-lengths 2 4 8 16 32 --device cuda --integration-workers 12
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-context-length")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from attribution.behavioral_responsibilities import checkpoint_model
from attribution.joint_linguistic_behavior_sensitivity import (
    DATA_DIR, LABELS, TEXTS, behavior_probabilities, training_statistics,
)
from attribution.regression_surprisal_analysis import (
    GeometryRegions, LINGUISTIC_PREDICTORS, PRIMARY, session_labels,
)
from spatio_temporal_reading.data.data import MecoDataset
from spatio_temporal_reading.data.dataLM import MecoDatasetLM


ANALYSIS_BEHAVIOURS = ("skip", "refixation", "regression")


def eligible_indices(frame: pd.DataFrame, inputs: torch.Tensor, feature_start: int, max_context: int) -> np.ndarray:
    """Indices whose target and immediately preceding input are usable.

    Output index ``n`` predicts raw fixation ``n`` from input index ``n``,
    which contains raw fixation ``n - 1``.  Hence ``n >= L`` is required for
    a context of L *actual preceding* fixations (rather than the BOS token).
    """
    valid = frame.behavioral_label.isin(PRIMARY) & frame.is_consecutive_original_fixation.eq(True)
    valid &= frame.index.to_series().ge(max_context).to_numpy()
    mask_offsets = [4, 5, 7]
    valid &= inputs[:, [feature_start + offset for offset in mask_offsets]].eq(0).all(dim=1).cpu().numpy()
    return np.flatnonzero(valid.to_numpy())


def contextual_variants(
    model: torch.nn.Module, inputs: torch.Tensor, targets: np.ndarray,
    context_length: int, feature_scales: list[tuple[int, float]],
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    contexts = torch.stack([inputs[index - context_length + 1:index + 1] for index in targets])
    count = len(targets)
    variants = contexts.repeat(2, 1, 1)
    for column, scale in feature_scales:
        variants[np.arange(count), -1, column] += scale
        variants[count + np.arange(count), -1, column] -= scale
    baseline = model(contexts)
    perturbed = model(variants)
    select = lambda result, rows: tuple(result[i][rows, -1].detach().cpu().numpy() for i in (0, 1, 3))
    return select(baseline, np.arange(count)), select(perturbed, np.arange(count)), select(perturbed, count + np.arange(count))


def bootstrap_summary(events: pd.DataFrame, behaviors: tuple[str, ...], replicates: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster bootstrap, retaining every context-length measurement per event."""
    lengths = sorted(events.context_length.unique())
    scanpaths, codes = np.unique(events.scanpath, return_inverse=True)
    # Events are written in context-length blocks, so align their weights by a
    # stable event key before applying the same sampled scanpaths at every L.
    event_columns = ["scanpath", "target_index"]
    wide = events.pivot(index=event_columns, columns="context_length")
    path_codes = pd.factorize(wide.index.get_level_values("scanpath"), sort=True)[0]
    n_paths = len(np.unique(path_codes))
    rng = np.random.default_rng(seed)
    boot_rows = []
    for replicate in tqdm(range(replicates), desc="Paired scanpath bootstrap"):
        counts = np.bincount(rng.integers(n_paths, size=n_paths), minlength=n_paths)
        weights = counts[path_codes]
        for behavior in behaviors:
            for length in lengths:
                signed = wide[(f"sensitivity_{behavior}", length)].to_numpy()
                boot_rows.append({"replicate": replicate, "behavior": behavior, "context_length": length,
                                  "statistic": "signed", "value": float(np.average(signed, weights=weights))})
                boot_rows.append({"replicate": replicate, "behavior": behavior, "context_length": length,
                                  "statistic": "absolute", "value": float(np.average(np.abs(signed), weights=weights))})
    bootstrap = pd.DataFrame(boot_rows)
    summary = (bootstrap.groupby(["behavior", "context_length", "statistic"], as_index=False)["value"]
               .agg(mean="mean", ci_lower=lambda x: x.quantile(.025), ci_upper=lambda x: x.quantile(.975)))
    # The point estimates should be empirical means, not bootstrap means.
    empirical = []
    for behavior in behaviors:
        for length in lengths:
            subset = events.loc[events.context_length.eq(length), f"sensitivity_{behavior}"]
            empirical.extend([
                {"behavior": behavior, "context_length": length, "statistic": "signed", "estimate": subset.mean()},
                {"behavior": behavior, "context_length": length, "statistic": "absolute", "estimate": subset.abs().mean()},
            ])
    summary = summary.merge(pd.DataFrame(empirical), on=["behavior", "context_length", "statistic"], validate="one_to_one")

    differences = []
    for behavior in behaviors:
        for statistic in ("signed", "absolute"):
            matrix = events.pivot(index=event_columns, columns="context_length", values=f"sensitivity_{behavior}")
            if statistic == "absolute":
                matrix = matrix.abs()
            for low, high in zip(lengths[:-1], lengths[1:]):
                observed = float((matrix[high] - matrix[low]).mean())
                samples = []
                # Reuse corresponding bootstrap replicates by calculating the
                # paired difference from the stored means, preserving pairing.
                low_boot = bootstrap.query("behavior == @behavior and statistic == @statistic and context_length == @low").sort_values("replicate").value.to_numpy()
                high_boot = bootstrap.query("behavior == @behavior and statistic == @statistic and context_length == @high").sort_values("replicate").value.to_numpy()
                samples = high_boot - low_boot
                differences.append({"behavior": behavior, "statistic": statistic, "low_context": low, "high_context": high,
                                    "estimate_difference": observed, "ci_lower": np.quantile(samples, .025), "ci_upper": np.quantile(samples, .975)})
            low, high = lengths[0], lengths[-1]
            observed = float((matrix[high] - matrix[low]).mean())
            low_boot = bootstrap.query("behavior == @behavior and statistic == @statistic and context_length == @low").sort_values("replicate").value.to_numpy()
            high_boot = bootstrap.query("behavior == @behavior and statistic == @statistic and context_length == @high").sort_values("replicate").value.to_numpy()
            samples = high_boot - low_boot
            differences.append({"behavior": behavior, "statistic": statistic, "low_context": low, "high_context": high,
                                "estimate_difference": observed, "ci_lower": np.quantile(samples, .025), "ci_upper": np.quantile(samples, .975)})
    return summary, pd.DataFrame(differences), bootstrap


def plot_summary(summary: pd.DataFrame, output: Path, behaviors: tuple[str, ...]) -> None:
    figure, axes = plt.subplots(2, len(behaviors), figsize=(4.5 * len(behaviors), 7), sharex="col", constrained_layout=True)
    for column, behavior in enumerate(behaviors):
        for row, statistic in enumerate(("signed", "absolute")):
            axis = axes[row, column]
            subset = summary.query("behavior == @behavior and statistic == @statistic").sort_values("context_length")
            axis.errorbar(subset.context_length, subset.estimate,
                          yerr=np.vstack([subset.estimate - subset.ci_lower, subset.ci_upper - subset.estimate]),
                          marker="o", color="#4C78A8", capsize=3)
            if statistic == "signed":
                axis.axhline(0, color="black", linewidth=.8)
                axis.set_title(behavior.replace("_", " ").title())
            axis.set_ylabel("Mean sensitivity" if statistic == "signed" else "Mean absolute sensitivity")
            axis.set_xlabel("Available preceding fixations (L)")
            axis.set_xticks(subset.context_length)
    figure.savefig(output / "context_length_linguistic_sensitivity.png", dpi=220)
    plt.close(figure)


def main(args: argparse.Namespace) -> None:
    lengths = sorted(set(args.context_lengths))
    if any(length < 1 for length in lengths):
        raise ValueError("Context lengths must be positive.")
    checkpoint = args.checkpoint_path or args.data / "saccade" / "TRANSFORMER_RAW_COV_LM" / "best_model.pt"
    output = (args.output or checkpoint.parent / "context_length_linguistic_sensitivity").resolve()
    output.mkdir(parents=True, exist_ok=True)
    model, uses_lm = checkpoint_model(checkpoint)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    model.to(device).eval()
    dataset_type = MecoDatasetLM if uses_lm else MecoDataset
    dataset = dataset_type(mode="test", filtering="raw", datadir=args.data)
    stats, correlations, _ = training_statistics(dataset_type, args.data, uses_lm)
    labels, texts = pd.read_csv(args.labels), pd.read_csv(args.texts)
    regions = GeometryRegions(texts, args.vertical_threshold, args.return_position_threshold, mode="word_box")
    feature_start = 4 + len(dataset.reader_to_idx)
    scales = [(feature_start + spec["input_offset"], stats[name]["sd"]) for name, spec in LINGUISTIC_PREDICTORS.items()]
    executor = ProcessPoolExecutor(max_workers=args.integration_workers, mp_context=multiprocessing.get_context("spawn")) if args.integration_workers > 1 else None
    events, processed = [], 0
    try:
        with torch.no_grad():
            for item_index, item in enumerate(tqdm(dataset.items, desc="Held-out scanpaths", disable=not args.progress)):
                text, reader = item[:2]
                inputs, _, _ = dataset[item_index]
                metadata = dataset.meco_df[(dataset.meco_df.text == text) & (dataset.meco_df.reader == reader)].copy()
                frame = session_labels(labels, metadata)
                indices = eligible_indices(frame, inputs, feature_start, max(lengths))
                if args.max_events is not None:
                    indices = indices[:max(0, args.max_events - processed)]
                if not len(indices):
                    continue
                inputs = inputs.to(device)
                for context_length in lengths:
                    for start in range(0, len(indices), args.perturbation_batch_size):
                        targets = indices[start:start + args.perturbation_batch_size]
                        base, high, low = contextual_variants(model, inputs, targets, context_length, scales)
                        payloads, rows = [], []
                        for local, target in enumerate(targets):
                            row = frame.iloc[target]
                            parameters = [tuple(values[local] for values in selected) for selected in (base, high, low)]
                            payloads.append((regions.rectangles(row), float(row.previous_x), float(row.previous_y), parameters))
                            rows.append((row, int(target)))
                        probabilities = list(executor.map(behavior_probabilities, payloads)) if executor else [behavior_probabilities(payload) for payload in payloads]
                        for (row, target), (original, plus, minus) in zip(rows, probabilities):
                            record = {"text": row.text, "reader": row.reader, "fixid": row.fixid, "target_index": target,
                                      "scanpath": f"{row.text}::{row.reader}", "context_length": context_length}
                            for behavior in PRIMARY:
                                record[f"probability_{behavior}"] = original[behavior]
                                record[f"probability_{behavior}_high"] = plus[behavior]
                                record[f"probability_{behavior}_low"] = minus[behavior]
                                record[f"sensitivity_{behavior}"] = (plus[behavior] - minus[behavior]) / 2
                            events.append(record)
                processed += len(indices)
                if args.max_events is not None and processed >= args.max_events:
                    break
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    if not events:
        raise RuntimeError("No eligible transitions found. Check labels, data, and requested maximum context length.")
    events = pd.DataFrame(events)
    summary, differences, bootstrap = bootstrap_summary(events, ANALYSIS_BEHAVIOURS, args.bootstrap_replicates, args.seed)
    events.to_csv(output / "context_length_linguistic_events.csv", index=False)
    summary.to_csv(output / "context_length_linguistic_summary.csv", index=False)
    differences.to_csv(output / "context_length_linguistic_paired_differences.csv", index=False)
    bootstrap.to_csv(output / "context_length_linguistic_bootstrap.csv", index=False)
    correlations.to_csv(output / "training_linguistic_correlations.csv")
    plot_summary(summary, output, ANALYSIS_BEHAVIOURS)
    metadata = {"checkpoint": str(Path(checkpoint).resolve()), "device": str(device), "context_lengths": lengths,
                "included_events_per_length": len(events) // len(lengths), "bootstrap_replicates": args.bootstrap_replicates,
                "geometry": "word_box", "perturbation": "joint +/- one training-set SD at fixation n-1 only",
                "training_feature_statistics": stats}
    (output / "summary.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--texts", type=Path, default=TEXTS)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    parser.add_argument("--vertical-threshold", type=float, default=.5)
    parser.add_argument("--return-position-threshold", type=float, default=.5)
    parser.add_argument("--perturbation-batch-size", type=int, default=16)
    parser.add_argument("--integration-workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", min(8, os.cpu_count() or 1))))
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-events", type=int)
    main(parser.parse_args())
