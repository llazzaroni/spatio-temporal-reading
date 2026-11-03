"""Grouped linguistic permutation test across available context lengths.

At every context length, each target is predicted from exactly L preceding
fixations.  The original inputs are compared with inputs in which the three
scalar linguistic values are jointly shuffled between compatible words.  A
positive result is a likelihood benefit (nats per predicted fixation) from
the intact linguistic information.

Unlike the behavioural analyses, this is an overall scanpath-likelihood test:
it deliberately uses all model-evaluable raw held-out targets, not only
word-aligned labelled transitions.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-context-permutation")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from attribution.behavioral_responsibilities import checkpoint_model
from attribution.linguistic_group_permutation import (
    DATA_DIR, log_likelihood, make_permutation_maps, permuted_inputs,
)
from spatio_temporal_reading.data.data import MecoDataset
from spatio_temporal_reading.data.dataLM import MecoDatasetLM


def contexts_at_targets(inputs: torch.Tensor, targets: np.ndarray, length: int) -> torch.Tensor:
    """Return L-input contexts ending at each target's preceding fixation."""
    return torch.stack([inputs[target - length + 1:target + 1] for target in targets])


def final_step_outputs(outputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    """Keep only the distribution predicting the target at each context end."""
    return tuple(value[:, -1] for value in outputs)


def bootstrap(scanpath_results: pd.DataFrame, replicates: int, seed: int, progress: bool) -> pd.DataFrame:
    """Nested paired bootstrap over scanpaths and linguistic permutations."""
    lengths = np.sort(scanpath_results.context_length.unique())
    permutations = np.sort(scanpath_results.permutation.unique())
    paths = scanpath_results[["text", "reader"]].drop_duplicates().reset_index(drop=True)
    lookup = scanpath_results.set_index(["permutation", "text", "reader", "context_length"])
    rng = np.random.default_rng(seed)
    draws = []
    for replicate in tqdm(range(replicates), desc="Paired nested bootstrap", disable=not progress):
        sampled_paths = paths.iloc[rng.integers(len(paths), size=len(paths))]
        sampled_permutations = rng.choice(permutations, size=len(permutations), replace=True)
        by_length = {}
        for length in lengths:
            values = []
            for permutation in sampled_permutations:
                selected = [lookup.loc[(permutation, row.text, row.reader, length)] for row in sampled_paths.itertuples(index=False)]
                values.append(sum(float(row.delta_log_likelihood) for row in selected) / sum(int(row.n_events) for row in selected))
            by_length[int(length)] = float(np.mean(values))
            draws.append({"replicate": replicate, "context_length": int(length), "likelihood_benefit_nats_per_target": by_length[int(length)]})
        for low, high in zip(lengths[:-1], lengths[1:]):
            draws.append({"replicate": replicate, "context_length": f"{low}->{high}",
                          "likelihood_benefit_nats_per_target": by_length[int(high)] - by_length[int(low)],
                          "kind": "adjacent_difference"})
        draws.append({"replicate": replicate, "context_length": f"{lengths[0]}->{lengths[-1]}",
                      "likelihood_benefit_nats_per_target": by_length[int(lengths[-1])] - by_length[int(lengths[0])],
                      "kind": "endpoint_difference"})
    return pd.DataFrame(draws)


def make_plot(summary: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    axis.errorbar(summary.context_length, summary.estimate,
                  yerr=np.vstack([summary.estimate - summary.ci_lower, summary.ci_upper - summary.estimate]),
                  marker="o", capsize=3, color="#4C78A8")
    axis.axhline(0, color="black", linewidth=.8)
    axis.set(xlabel="Available preceding fixations (L)", ylabel="Original − permuted likelihood\n(nats per predicted fixation)",
             title="Grouped linguistic-feature reliance by context length")
    axis.set_xticks(summary.context_length)
    figure.savefig(output / "context_length_linguistic_permutation.png", dpi=220)
    plt.close(figure)


def main(args: argparse.Namespace) -> None:
    lengths = sorted(set(args.context_lengths))
    if not lengths or min(lengths) < 1:
        raise ValueError("--context-lengths must contain positive integers.")
    checkpoint = args.checkpoint_path or args.data / "saccade" / "TRANSFORMER_RAW_COV_LM" / "best_model.pt"
    output = (args.output or checkpoint.parent / "context_length_linguistic_permutation").resolve()
    output.mkdir(parents=True, exist_ok=True)
    model, uses_lm = checkpoint_model(checkpoint)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is not available.")
    model.to(device).eval()
    dataset_type = MecoDatasetLM if uses_lm else MecoDataset
    dataset = dataset_type(mode="test", filtering="raw", datadir=args.data)
    test_texts = sorted({int(item[0]) for item in dataset.items})
    maps, diagnostics = make_permutation_maps(dataset.meco_df, test_texts, args.permutations, args.seed, args.length_matching)
    feature_start = 4 + len(dataset.reader_to_idx)
    rows = []

    with torch.no_grad():
        for item_index, item in enumerate(tqdm(dataset.items, desc="Held-out scanpaths", unit="scanpath", disable=not args.progress)):
            if args.max_scanpaths is not None and item_index >= args.max_scanpaths:
                break
            text, reader = item[:2]
            inputs, positions, saccades = dataset[item_index]
            inputs, positions, saccades = inputs.to(device), positions.to(device), saccades.to(device)
            targets = np.arange(max(lengths), len(inputs))
            if args.max_targets is not None:
                targets = targets[:args.max_targets]
            if not len(targets):
                continue
            subset = dataset.meco_df[(dataset.meco_df.text == text) & (dataset.meco_df.reader == reader)].copy()
            permuted = [permuted_inputs(inputs, subset, int(text), mapping, feature_start)[0] for mapping in maps]
            for length in lengths:
                baseline_total = 0.0
                permuted_totals = np.zeros(args.permutations, dtype=float)
                for start in range(0, len(targets), args.target_batch_size):
                    batch_targets = targets[start:start + args.target_batch_size]
                    target_positions, target_saccades = positions[batch_targets], saccades[batch_targets]
                    original_contexts = contexts_at_targets(inputs, batch_targets, length)
                    baseline_total += float(log_likelihood(final_step_outputs(model(original_contexts)), target_positions, target_saccades).sum().cpu())
                    for perm_start in range(0, args.permutations, args.permutation_batch_size):
                        permutation_ids = range(perm_start, min(args.permutations, perm_start + args.permutation_batch_size))
                        variants = torch.cat([contexts_at_targets(permuted[permutation], batch_targets, length) for permutation in permutation_ids])
                        repeats = len(permutation_ids)
                        ll = log_likelihood(
                            final_step_outputs(model(variants)),
                            target_positions.unsqueeze(0).expand(repeats, -1, -1).reshape(-1, 2),
                            target_saccades.unsqueeze(0).expand(repeats, -1).reshape(-1),
                        ).reshape(repeats, len(batch_targets)).sum(dim=1).cpu().numpy()
                        permuted_totals[list(permutation_ids)] += ll
                for permutation, permuted_total in enumerate(permuted_totals):
                    rows.append({"permutation": permutation, "text": text, "reader": reader, "context_length": length,
                                 "n_events": len(targets), "baseline_log_likelihood": baseline_total,
                                 "permuted_log_likelihood": float(permuted_total),
                                 "delta_log_likelihood": baseline_total - float(permuted_total),
                                 "delta_nats_per_target": (baseline_total - float(permuted_total)) / len(targets)})

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("No targets remained. Check context lengths and held-out scanpath lengths.")
    boot = bootstrap(results, args.bootstrap_replicates, args.seed + 1, args.progress)
    main_boot = boot[boot.get("kind").isna()].copy()
    summary_rows = []
    for length in lengths:
        subset = results.loc[results.context_length.eq(length)]
        values = main_boot.loc[main_boot.context_length.eq(length), "likelihood_benefit_nats_per_target"]
        summary_rows.append({"context_length": length,
                             "estimate": subset.delta_log_likelihood.sum() / subset.n_events.sum(),
                             "ci_lower": values.quantile(.025), "ci_upper": values.quantile(.975),
                             "events_per_permutation": int(subset.n_events.sum() / args.permutations)})
    summary = pd.DataFrame(summary_rows)
    differences = boot[boot.get("kind").notna()].groupby(["kind", "context_length"], as_index=False)["likelihood_benefit_nats_per_target"].agg(
        estimate="mean", ci_lower=lambda x: x.quantile(.025), ci_upper=lambda x: x.quantile(.975)
    )
    estimate_by_length = summary.set_index("context_length").estimate
    for index, row in differences.iterrows():
        low, high = map(int, str(row.context_length).split("->"))
        differences.loc[index, "estimate"] = estimate_by_length[high] - estimate_by_length[low]
    results.to_csv(output / "context_length_permutation_scanpath_likelihoods.csv", index=False)
    summary.to_csv(output / "context_length_permutation_summary.csv", index=False)
    differences.to_csv(output / "context_length_permutation_paired_differences.csv", index=False)
    boot.to_csv(output / "context_length_permutation_bootstrap.csv", index=False)
    diagnostics.to_csv(output / "permutation_matching_diagnostics.csv", index=False)
    make_plot(summary, output)
    metadata = {"checkpoint": str(Path(checkpoint).resolve()), "device": str(device), "context_lengths": lengths,
                "permutations": args.permutations, "length_matching": args.length_matching,
                "targets": "all raw model-evaluable test targets with at least max(L) preceding fixations",
                "permutation": "joint within-text shuffle of character surprisal, word surprisal, and frequency; masks preserved",
                "interpretation": "A larger value at longer L can reflect both stronger use of history and more available linguistic inputs."}
    (output / "summary.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    parser.add_argument("--permutations", type=int, default=25)
    parser.add_argument("--permutation-batch-size", type=int, default=8)
    parser.add_argument("--target-batch-size", type=int, default=64)
    parser.add_argument("--length-matching", choices=["exact", "none"], default="exact")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-scanpaths", type=int)
    parser.add_argument("--max-targets", type=int, help="Debug option: targets per scanpath.")
    main(parser.parse_args())
