"""Compare full and no-scalar-linguistic COV+LM models on held-out likelihood."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.distributions import LogNormal, MultivariateNormal
from tqdm.auto import tqdm

from attribution.behavioral_responsibilities import checkpoint_model
from spatio_temporal_reading.data.dataLM import MecoDatasetLM, MecoDatasetLMNoLinguistic


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def joint_log_likelihood(outputs, positions, saccades):
    weights, means, saccade_means, covariances, saccade_scales = outputs
    terms = (
        torch.log(weights.clamp_min(1e-12))
        + MultivariateNormal(means, covariances).log_prob(positions.unsqueeze(-2))
        + LogNormal(saccade_means, saccade_scales).log_prob(saccades.unsqueeze(-1))
    )
    return torch.logsumexp(terms, dim=-1)


def checkpoint_ablation_flag(path: Path) -> bool:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    config = config if isinstance(config, dict) else vars(config)
    return bool(config.get("ablate_linguistic_features", False))


def main(args):
    full_path, ablation_path = args.full_checkpoint.resolve(), args.ablation_checkpoint.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if checkpoint_ablation_flag(full_path):
        raise ValueError("--full-checkpoint is marked as an ablation checkpoint.")
    if not checkpoint_ablation_flag(ablation_path):
        raise ValueError("--ablation-checkpoint lacks ablation metadata; use the checkpoint produced by --ablate-linguistic-features.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    full_model, full_lm = checkpoint_model(full_path)
    ablation_model, ablation_lm = checkpoint_model(ablation_path)
    if not (full_lm and ablation_lm):
        raise ValueError("This comparison is for the matched LM models.")
    full_model.to(device).eval()
    ablation_model.to(device).eval()
    full_ds = MecoDatasetLM(mode="test", filtering=args.filtering, datadir=args.data)
    ablation_ds = MecoDatasetLMNoLinguistic(mode="test", filtering=args.filtering, datadir=args.data)
    if full_ds.items != ablation_ds.items:
        raise RuntimeError("Full and ablation test splits differ.")
    rows = []
    with torch.no_grad():
        for index, item in enumerate(tqdm(full_ds.items, desc="Held-out scanpaths", disable=not args.progress)):
            full_input, positions, saccades = full_ds[index]
            ablation_input, _, _ = ablation_ds[index]
            positions, saccades = positions.to(device), saccades.to(device)
            ll_full = joint_log_likelihood(full_model(full_input.unsqueeze(0).to(device)), positions.unsqueeze(0), saccades.unsqueeze(0)).sum().item()
            ll_ablation = joint_log_likelihood(ablation_model(ablation_input.unsqueeze(0).to(device)), positions.unsqueeze(0), saccades.unsqueeze(0)).sum().item()
            rows.append({"text": item[0], "reader": item[1], "n_events": len(positions), "full_log_likelihood": ll_full, "ablation_log_likelihood": ll_ablation, "delta_log_likelihood": ll_full - ll_ablation, "delta_nats_per_fixation": (ll_full - ll_ablation) / len(positions)})
    result = pd.DataFrame(rows)
    rng = np.random.default_rng(args.seed)
    bootstrap = []
    for _ in range(args.bootstrap_replicates):
        sample = result.iloc[rng.integers(len(result), size=len(result))]
        bootstrap.append(sample["delta_log_likelihood"].sum() / sample["n_events"].sum())
    bootstrap = np.asarray(bootstrap)
    summary = {
        "full_checkpoint": str(full_path), "ablation_checkpoint": str(ablation_path), "device": str(device),
        "held_out_scanpaths": len(result), "held_out_events": int(result.n_events.sum()),
        "delta_nats_per_fixation": float(result.delta_log_likelihood.sum() / result.n_events.sum()),
        "scanpath_bootstrap_ci": np.quantile(bootstrap, [.025, .975]).tolist(),
    }
    result.to_csv(output / "heldout_likelihood_comparison.csv", index=False)
    pd.DataFrame({"delta_nats_per_fixation": bootstrap}).to_csv(output / "scanpath_bootstrap.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].hist(result.delta_nats_per_fixation, bins=20, color="#4C78A8")
    axes[0].axvline(0, color="black")
    axes[0].set(title="Per-scanpath held-out likelihood benefit", xlabel="Full minus ablation (nats / fixation)", ylabel="Scanpaths")
    axes[1].hist(bootstrap, bins=30, color="#54A24B")
    axes[1].axvline(0, color="black")
    axes[1].set(title="Scanpath bootstrap", xlabel="Mean likelihood benefit (nats / fixation)", ylabel="Bootstrap draws")
    figure.savefig(output / "linguistic_ablation_likelihood_comparison.png", dpi=220)
    plt.close(figure)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--filtering", choices=["raw", "filtered"], default="raw")
    parser.add_argument("--full-checkpoint", type=Path, required=True)
    parser.add_argument("--ablation-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    main(parser.parse_args())
