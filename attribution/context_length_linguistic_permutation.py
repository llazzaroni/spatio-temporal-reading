"""Grouped linguistic permutation test over natural scanpath-history bands.

By default, endpoints ``2 4 8 16`` define disjoint prediction-index bands
``[0,2), [2,4), [4,8), [8,16)``.  The model receives its normal full causal
prefix at every prediction; no artificial windowing or sequence cropping occurs.
Use ``--aggregation prefix`` to reproduce cumulative-prefix averages instead.
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
from attribution.linguistic_group_permutation import DATA_DIR, log_likelihood, make_permutation_maps, permuted_inputs
from spatio_temporal_reading.data.data import MecoDataset
from spatio_temporal_reading.data.dataLM import MecoDatasetLM


def bootstrap(results: pd.DataFrame, replicates: int, seed: int, progress: bool) -> pd.DataFrame:
    """Paired nested bootstrap over held-out scanpaths and permutation IDs."""
    lengths = np.sort(results.context_length.unique())
    permutations = np.sort(results.permutation.unique())
    paths = results[["text", "reader"]].drop_duplicates().reset_index(drop=True)
    lookup = results.set_index(["permutation", "text", "reader", "context_length"])
    rng = np.random.default_rng(seed)
    draws = []
    for replicate in tqdm(range(replicates), desc="Paired nested bootstrap", disable=not progress):
        sampled_paths = paths.iloc[rng.integers(len(paths), size=len(paths))]
        sampled_permutations = rng.choice(permutations, size=len(permutations), replace=True)
        estimates = {}
        for length in lengths:
            by_permutation = []
            for permutation in sampled_permutations:
                rows = [lookup.loc[(permutation, path.text, path.reader, length)] for path in sampled_paths.itertuples(index=False)]
                by_permutation.append(sum(float(row.delta_log_likelihood) for row in rows) / sum(int(row.n_events) for row in rows))
            estimates[int(length)] = float(np.mean(by_permutation))
            draws.append({"replicate": replicate, "context_length": int(length), "kind": "estimate", "value": estimates[int(length)]})
        for low, high in zip(lengths[:-1], lengths[1:]):
            draws.append({"replicate": replicate, "context_length": f"{low}->{high}", "kind": "adjacent_difference", "value": estimates[int(high)] - estimates[int(low)]})
        draws.append({"replicate": replicate, "context_length": f"{lengths[0]}->{lengths[-1]}", "kind": "endpoint_difference", "value": estimates[int(lengths[-1])] - estimates[int(lengths[0])]})
    return pd.DataFrame(draws)


def save_plot(summary: pd.DataFrame, output: Path, aggregation: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    x = summary.x_position
    ax.errorbar(x, summary.estimate,
                yerr=np.vstack([summary.estimate - summary.ci_lower, summary.ci_upper - summary.estimate]),
                marker="o", capsize=3, color="#4C78A8")
    ax.axhline(0, color="black", linewidth=.8)
    xlabel = "Scanpath prefix length (predicted fixations)" if aggregation == "prefix" else "Prediction-index band (available history increases left to right)"
    title = "Grouped linguistic-feature benefit over scanpath prefixes" if aggregation == "prefix" else "Grouped linguistic-feature benefit by scanpath-history band"
    ax.set(xlabel=xlabel,
           ylabel="Original − permuted likelihood\n(nats per predicted fixation)",
           title=title)
    ax.set_xticks(x)
    if aggregation == "interval":
        ax.set_xticklabels([f"{int(start)}–{int(end - 1)}" for start, end in zip(summary.interval_start, summary.interval_end)])
    fig.savefig(output / "context_length_linguistic_permutation.png", dpi=220)
    plt.close(fig)


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
    rows, excluded_short = [], 0

    with torch.no_grad():
        for item_index, item in enumerate(tqdm(dataset.items, desc="Held-out scanpaths", unit="scanpath", disable=not args.progress)):
            if args.max_scanpaths is not None and item_index >= args.max_scanpaths:
                break
            text, reader = item[:2]
            inputs, positions, saccades = dataset[item_index]
            if len(inputs) < max(lengths):
                excluded_short += 1
                continue
            inputs, positions, saccades = inputs.to(device), positions.to(device), saccades.to(device)
            subset = dataset.meco_df[(dataset.meco_df.text == text) & (dataset.meco_df.reader == reader)].copy()
            baseline_ll = log_likelihood(model(inputs.unsqueeze(0)), positions.unsqueeze(0), saccades.unsqueeze(0)).squeeze(0)
            for start in range(0, args.permutations, args.permutation_batch_size):
                permutation_ids = range(start, min(args.permutations, start + args.permutation_batch_size))
                variants = [permuted_inputs(inputs, subset, int(text), maps[p], feature_start)[0] for p in permutation_ids]
                variant_ll = log_likelihood(model(torch.stack(variants)), positions.unsqueeze(0).expand(len(variants), -1, -1), saccades.unsqueeze(0).expand(len(variants), -1))
                for local, permutation in enumerate(permutation_ids):
                    for index, length in enumerate(lengths):
                        interval_start = 0 if args.aggregation == "prefix" or index == 0 else lengths[index - 1]
                        baseline_slice = baseline_ll[interval_start:length].sum()
                        permuted_slice = variant_ll[local, interval_start:length].sum()
                        delta = float((baseline_slice - permuted_slice).cpu())
                        rows.append({"permutation": permutation, "text": text, "reader": reader, "context_length": length,
                                     "interval_start": interval_start, "interval_end": length,
                                     "n_events": length - interval_start, "baseline_log_likelihood": float(baseline_slice.cpu()),
                                     "permuted_log_likelihood": float(permuted_slice.cpu()), "delta_log_likelihood": delta,
                                     "delta_nats_per_target": delta / (length - interval_start)})
    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("No scanpaths were at least as long as the largest requested prefix.")
    draws = bootstrap(results, args.bootstrap_replicates, args.seed + 1, args.progress)
    summary_rows = []
    for length in lengths:
        subset = results[results.context_length.eq(length)]
        values = draws[(draws.kind == "estimate") & (draws.context_length == length)].value
        interval_start = 0 if args.aggregation == "prefix" or lengths.index(length) == 0 else lengths[lengths.index(length) - 1]
        summary_rows.append({"context_length": length, "estimate": subset.delta_log_likelihood.sum() / subset.n_events.sum(),
                             "ci_lower": values.quantile(.025), "ci_upper": values.quantile(.975),
                             "interval_start": interval_start, "interval_end": length,
                             "x_position": length if args.aggregation == "prefix" else (interval_start + length) / 2,
                             "scanpaths": subset[["text", "reader"]].drop_duplicates().shape[0],
                             "events_per_permutation": int(subset.n_events.sum() / args.permutations)})
    summary = pd.DataFrame(summary_rows)
    differences = draws[draws.kind.ne("estimate")].groupby(["kind", "context_length"], as_index=False).value.agg(
        estimate="mean", ci_lower=lambda x: x.quantile(.025), ci_upper=lambda x: x.quantile(.975))
    point = summary.set_index("context_length").estimate
    for i, row in differences.iterrows():
        low, high = map(int, row.context_length.split("->"))
        differences.loc[i, "estimate"] = point[high] - point[low]
    results.to_csv(output / "context_length_permutation_scanpath_likelihoods.csv", index=False)
    summary.to_csv(output / "context_length_permutation_summary.csv", index=False)
    differences.to_csv(output / "context_length_permutation_paired_differences.csv", index=False)
    draws.to_csv(output / "context_length_permutation_bootstrap.csv", index=False)
    diagnostics.to_csv(output / "permutation_matching_diagnostics.csv", index=False)
    save_plot(summary, output, args.aggregation)
    metadata = {"checkpoint": str(Path(checkpoint).resolve()), "device": str(device), "context_lengths": lengths,
                "permutations": args.permutations, "length_matching": args.length_matching,
                "scanpaths_excluded_for_short_length": excluded_short,
                "aggregation": args.aggregation,
                "targets": "first L raw model-evaluable targets" if args.aggregation == "prefix" else "disjoint bands [previous endpoint, current endpoint) of raw model-evaluable targets",
                "permutation": "same grouped within-text linguistic-triplet permutation as the original experiment",
                "model_context": "ordinary full causal prefix; no cropping or masking"}
    (output / "summary.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    parser.add_argument("--aggregation", choices=["interval", "prefix"], default="interval", help="Default: disjoint bands [previous endpoint, endpoint).")
    parser.add_argument("--permutations", type=int, default=25)
    parser.add_argument("--permutation-batch-size", type=int, default=8)
    parser.add_argument("--length-matching", choices=["exact", "none"], default="exact")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261029, help="Matches the original permutation experiment.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-scanpaths", type=int)
    main(parser.parse_args())
