"""Grouped test-time permutation test for scalar linguistic-input reliance.

The three scalar linguistic inputs (character surprisal, word surprisal, and
lexical rarity) are shuffled jointly between compatible words within a text.
The model and all eye-movement history remain fixed.  A positive likelihood
difference means the original linguistic triplets helped the trained model
assign greater probability to the held-out scanpaths.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-linguistic-permutation")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import multivariate_normal
from torch.distributions import LogNormal, MultivariateNormal
from tqdm.auto import tqdm

from attribution.behavioral_responsibilities import checkpoint_model
from spatio_temporal_reading.data.data import MecoDataset
from spatio_temporal_reading.data.dataLM import MecoDatasetLM


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
LINGUISTIC_COLUMNS = ("char_level_surp", "word_level_surprisal", "freq")
LINGUISTIC_MASKS = ("char_level_surp_nan", "word_level_surprisal_nan", "freq_nan")
FEATURE_OFFSETS = (0, 1, 3)  # Positions inside the eight scalar feature channels.


def log_likelihood(outputs, positions: torch.Tensor, saccades: torch.Tensor) -> torch.Tensor:
    """Joint spatial-and-temporal mixture log likelihood for each target."""
    weights, means, saccade_means, covariances, saccade_scales = outputs
    log_weights = torch.log(weights.clamp_min(1e-12))
    log_spatial = MultivariateNormal(means, covariances).log_prob(positions.unsqueeze(-2))
    log_temporal = LogNormal(saccade_means, saccade_scales).log_prob(saccades.unsqueeze(-1))
    return torch.logsumexp(log_weights + log_spatial + log_temporal, dim=-1)


def word_inventory(meco: pd.DataFrame, text: int, length_matching: str) -> pd.DataFrame:
    """One scalar-feature record per text word, in model-input representation."""
    columns = ["text", "ianum_word", "len", *LINGUISTIC_COLUMNS, *LINGUISTIC_MASKS]
    words = meco.loc[(meco["text"] == text) & meco["ianum_word"].notna(), columns].copy()
    words = words.groupby(["text", "ianum_word"], as_index=False).first()
    words["mask_pattern"] = list(map(tuple, words.loc[:, LINGUISTIC_MASKS].astype(int).to_numpy()))
    words["length_stratum"] = words["len"].astype(float) if length_matching == "exact" else 0
    return words


def make_permutation_maps(meco: pd.DataFrame, texts: list[int], permutations: int, seed: int, length_matching: str):
    """Return one donor-word map per permutation and matching diagnostics."""
    rng = np.random.default_rng(seed)
    inventories = {int(text): word_inventory(meco, int(text), length_matching) for text in texts}
    maps, diagnostics = [], []
    for permutation in range(permutations):
        mapping: dict[tuple[int, int], np.ndarray] = {}
        eligible = moved = 0
        for text, words in inventories.items():
            for _, stratum in words.groupby(["mask_pattern", "length_stratum"], sort=False):
                indices = stratum.index.to_numpy()
                donors = rng.permutation(indices)
                eligible += len(indices)
                moved += int(np.sum(indices != donors))
                for recipient, donor in zip(indices, donors):
                    recipient_key = (text, int(words.loc[recipient, "ianum_word"]))
                    mapping[recipient_key] = words.loc[donor, list(LINGUISTIC_COLUMNS)].to_numpy(dtype=np.float32)
        maps.append(mapping)
        diagnostics.append({"permutation": permutation, "eligible_words": eligible, "moved_words": moved, "moved_word_fraction": moved / eligible if eligible else np.nan})
    return maps, pd.DataFrame(diagnostics)


def permuted_inputs(inputs: torch.Tensor, subset: pd.DataFrame, text: int, donor_map: dict[tuple[int, int], np.ndarray], feature_start: int) -> tuple[torch.Tensor, int]:
    """Replace only the previous-fixation linguistic triplet at input tokens."""
    result = inputs.clone()
    changed = 0
    # Input position j+1 contains the covariates of raw fixation j; position 0
    # is BOS and must never be changed.
    for fixation_index, word in enumerate(subset["ianum_word"].to_numpy()):
        if pd.isna(word):
            continue
        donor_values = donor_map.get((int(text), int(word)))
        if donor_values is None:
            continue
        position = fixation_index + 1
        if position >= len(result):  # Last observed fixation has no next target.
            continue
        columns = [feature_start + offset for offset in FEATURE_OFFSETS]
        old = result[position, columns]
        new = torch.as_tensor(donor_values, device=result.device, dtype=result.dtype)
        changed += int(not torch.equal(old, new))
        result[position, columns] = new
    return result, changed


def bootstrap_scanpaths(scanpath_results: pd.DataFrame, replicates: int, seed: int, progress: bool) -> np.ndarray:
    """Nested bootstrap over held-out scanpaths and permutation replicates."""
    rng = np.random.default_rng(seed)
    permutations = np.sort(scanpath_results["permutation"].unique())
    scanpaths = scanpath_results[["text", "reader"]].drop_duplicates().reset_index(drop=True)
    lookup = scanpath_results.set_index(["permutation", "text", "reader"])
    draws = []
    for _ in tqdm(range(replicates), desc="Nested scanpath/permutation bootstrap", disable=not progress):
        sampled_paths = scanpaths.iloc[rng.integers(len(scanpaths), size=len(scanpaths))]
        sampled_permutations = rng.choice(permutations, size=len(permutations), replace=True)
        values = []
        for permutation in sampled_permutations:
            rows = [lookup.loc[(permutation, row.text, row.reader)] for row in sampled_paths.itertuples(index=False)]
            total_delta = sum(float(row.delta_log_likelihood) for row in rows)
            total_events = sum(int(row.n_events) for row in rows)
            values.append(total_delta / total_events)
        draws.append(float(np.mean(values)))
    return np.asarray(draws)


def save_figure(permutation_summary: pd.DataFrame, bootstrap: np.ndarray, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].hist(permutation_summary["delta_nats_per_fixation"], bins=min(20, len(permutation_summary)), color="#4C78A8")
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].set(title="Grouped linguistic permutations", xlabel="Original minus permuted nats / fixation", ylabel="Permutations")
    axes[1].hist(bootstrap, bins=30, color="#54A24B")
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set(title="Nested scanpath/permutation bootstrap", xlabel="Mean likelihood benefit (nats / fixation)", ylabel="Bootstrap draws")
    figure.savefig(output / "linguistic_group_permutation_likelihood.png", dpi=220)
    plt.close(figure)


def main(args: argparse.Namespace) -> None:
    checkpoint = args.checkpoint_path or args.data / "saccade" / "TRANSFORMER_RAW_COV_LM" / "best_model.pt"
    output = (args.output or checkpoint.parent / "linguistic_group_permutation").resolve()
    output.mkdir(parents=True, exist_ok=True)
    model, uses_lm = checkpoint_model(checkpoint)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is not available.")
    model.to(device)
    dataset_type = MecoDatasetLM if uses_lm else MecoDataset
    dataset = dataset_type(mode="test", filtering=args.filtering, datadir=args.data)
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
            subset = dataset.meco_df[(dataset.meco_df["text"] == text) & (dataset.meco_df["reader"] == reader)].copy()
            baseline_ll = log_likelihood(model(inputs.unsqueeze(0)), positions.unsqueeze(0), saccades.unsqueeze(0)).squeeze(0)
            for start in range(0, args.permutations, args.permutation_batch_size):
                permutation_ids = range(start, min(args.permutations, start + args.permutation_batch_size))
                variants, changed = [], []
                for permutation in permutation_ids:
                    variant, count = permuted_inputs(inputs, subset, int(text), maps[permutation], feature_start)
                    variants.append(variant)
                    changed.append(count)
                batch = torch.stack(variants)
                variant_ll = log_likelihood(model(batch), positions.unsqueeze(0).expand(len(variants), -1, -1), saccades.unsqueeze(0).expand(len(variants), -1))
                for local, permutation in enumerate(permutation_ids):
                    delta = float((baseline_ll - variant_ll[local]).sum().cpu())
                    rows.append({"permutation": permutation, "text": text, "reader": reader, "n_events": len(baseline_ll), "baseline_log_likelihood": float(baseline_ll.sum().cpu()), "permuted_log_likelihood": float(variant_ll[local].sum().cpu()), "delta_log_likelihood": delta, "delta_nats_per_fixation": delta / len(baseline_ll), "changed_input_positions": changed[local]})

    scanpath_results = pd.DataFrame(rows)
    permutation_summary = scanpath_results.groupby("permutation", as_index=False).agg(
        total_events=("n_events", "sum"), total_delta_log_likelihood=("delta_log_likelihood", "sum"), changed_input_positions=("changed_input_positions", "sum")
    )
    permutation_summary["delta_nats_per_fixation"] = permutation_summary["total_delta_log_likelihood"] / permutation_summary["total_events"]
    bootstrap = bootstrap_scanpaths(scanpath_results, args.bootstrap_replicates, args.seed + 1, args.progress)
    summary = {
        "checkpoint": str(Path(checkpoint).resolve()), "uses_lm_embeddings": uses_lm, "device": str(device),
        "filtering": args.filtering, "permutations": args.permutations, "length_matching": args.length_matching,
        "mean_delta_nats_per_fixation": float(permutation_summary["delta_nats_per_fixation"].mean()),
        "permutation_range": np.quantile(permutation_summary["delta_nats_per_fixation"], [.025, .975]).tolist(),
        "nested_bootstrap_ci": np.quantile(bootstrap, [.025, .975]).tolist(),
        "mean_moved_word_fraction": float(diagnostics["moved_word_fraction"].mean()),
    }
    diagnostics.to_csv(output / "permutation_matching_diagnostics.csv", index=False)
    scanpath_results.to_csv(output / "permutation_scanpath_likelihoods.csv", index=False)
    permutation_summary.to_csv(output / "permutation_likelihood_summary.csv", index=False)
    pd.DataFrame({"mean_delta_nats_per_fixation": bootstrap}).to_csv(output / "nested_bootstrap.csv", index=False)
    save_figure(permutation_summary, bootstrap, output)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote results to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--filtering", choices=["raw", "filtered"], default="raw")
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--permutations", type=int, default=25)
    parser.add_argument("--permutation-batch-size", type=int, default=8)
    parser.add_argument("--length-matching", choices=["exact", "none"], default="exact")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-scanpaths", type=int, help="Debug option: limit held-out scanpaths.")
    main(parser.parse_args())
