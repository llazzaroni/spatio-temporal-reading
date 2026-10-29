"""Associate saccade-mixture components with behavioural transition labels.

The analysis is for covariance-aware saccade Transformers.  For every held-out
transition it computes posterior mixture responsibilities

    r[n, k] = w[n, k] p_k(time[n], position[n]) / sum_j w[n, j] p_j(...)

and their spatial-only analogue.  It then writes component-by-behaviour
association and lift matrices.

Example (the default final filtered COV+LM checkpoint)::

    python -m attribution.behavioral_responsibilities

Use ``--filtering raw`` to analyse the raw model. Raw analyses automatically
retain only consecutive original word-to-word transitions, so the behavioural
label and raw-model target always describe the same movement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.distributions import LogNormal, MultivariateNormal
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from spatio_temporal_reading.data.data import MecoDataset
from spatio_temporal_reading.data.dataLM import MecoDatasetLM
from spatio_temporal_reading.model.model import TransformerCov
from spatio_temporal_reading.model.modelLM import TransformerCovLM


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_LABELS = DATA_DIR / "hp_augmented_meco_100_1000_1_10_model_filtered_behavioral_labels.csv"
PRIMARY_BEHAVIOURS = ["forward", "skip", "refixation", "regression", "line_return"]


def checkpoint_model(checkpoint_path: Path) -> tuple[torch.nn.Module, bool]:
    """Load a COV saccade model and return it with its LM-input status."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if not isinstance(config, dict):
        config = vars(config).copy()
    # Training metadata is retained in new ablation checkpoints but is not a
    # Transformer constructor argument.
    config.pop("ablate_linguistic_features", None)
    config.pop("seed", None)
    config.setdefault("dropout", 0.0)
    state_dict = checkpoint["model_state_dict"]
    is_lm = any(key.startswith("gptProjector.") for key in state_dict)
    model = TransformerCovLM(**config) if is_lm else TransformerCov(**config)
    model.load_state_dict(state_dict)
    model.eval()
    return model, is_lm


def log_responsibilities(
    weights: torch.Tensor,
    position_means: torch.Tensor,
    saccade_means: torch.Tensor,
    covariances: torch.Tensor,
    saccade_scales: torch.Tensor,
    positions: torch.Tensor,
    saccades: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return joint responsibility, spatial responsibility, and prior weights."""
    log_weights = torch.log(weights.clamp_min(1e-12))
    log_spatial = MultivariateNormal(position_means, covariances).log_prob(
        positions.unsqueeze(2)
    )
    log_temporal = LogNormal(saccade_means, saccade_scales).log_prob(
        saccades.unsqueeze(2)
    )
    joint = torch.softmax(log_weights + log_spatial + log_temporal, dim=-1)
    spatial = torch.softmax(log_weights + log_spatial, dim=-1)
    return joint, spatial, weights


def association_matrix(
    responsibilities: np.ndarray, labels: pd.Series, behaviours: list[str]
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Compute A[k,b], empirical behaviour frequencies, and association lift."""
    one_hot = np.column_stack([(labels == behaviour).to_numpy(float) for behaviour in behaviours])
    component_mass = responsibilities.sum(axis=0)
    association = responsibilities.T @ one_hot / component_mass[:, None]
    frequencies = pd.Series(one_hot.mean(axis=0), index=behaviours, name="baseline_frequency")
    components = [f"component_{index}" for index in range(responsibilities.shape[1])]
    association_df = pd.DataFrame(association, index=components, columns=behaviours)
    lift_df = association_df.div(frequencies, axis="columns")
    return association_df, frequencies, lift_df


def component_support(responsibilities: np.ndarray) -> pd.DataFrame:
    """Summarise posterior mass and effective sample size for each component."""
    total_mass = responsibilities.sum(axis=0)
    squared_mass = np.square(responsibilities).sum(axis=0)
    effective_n = np.divide(
        np.square(total_mass), squared_mass,
        out=np.zeros_like(total_mass), where=squared_mass > 0,
    )
    return pd.DataFrame(
        {
            "total_responsibility": total_mass,
            "mean_responsibility": responsibilities.mean(axis=0),
            "effective_sample_size": effective_n,
        },
        index=[f"component_{index}" for index in range(responsibilities.shape[1])],
    )


def save_heatmap(matrix: pd.DataFrame, path: Path, title: str, *, lift: bool) -> None:
    """Save a labelled component-by-behaviour heatmap."""
    values = matrix.to_numpy()
    figure, axis = plt.subplots(figsize=(8, 8), constrained_layout=True)
    if lift:
        vmax = max(2.0, float(np.nanmax(values)))
        image = axis.imshow(
            values,
            aspect="auto",
            cmap="coolwarm",
            norm=TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=vmax),
        )
        value_format = ".2f"
    else:
        image = axis.imshow(values, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        value_format = ".2f"
    axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=30, ha="right")
    axis.set_yticks(range(len(matrix.index)), matrix.index)
    axis.set_xlabel("Behaviour")
    axis.set_ylabel("Mixture component")
    axis.set_title(title)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            colour = "white" if (not lift and values[row, column] > .5) else "black"
            axis.text(column, row, format(values[row, column], value_format), ha="center", va="center", fontsize=7, color=colour)
    figure.colorbar(image, ax=axis, label="Lift relative to behavioural baseline" if lift else "Association probability")
    figure.savefig(path, dpi=220)
    plt.close(figure)


def save_joint_lift_with_utilization(
    lift: pd.DataFrame,
    prior_support: pd.DataFrame,
    joint_support: pd.DataFrame,
    path: Path,
) -> None:
    """Plot joint lift with prior/posterior usage and posterior support."""
    components = lift.index
    values = lift.to_numpy()
    figure, axes = plt.subplots(
        1, 3,
        figsize=(13, 8),
        gridspec_kw={"width_ratios": [5.0, 2.6, 2.0]},
        constrained_layout=True,
    )
    vmax = max(2.0, float(np.nanmax(values)))
    image = axes[0].imshow(
        values,
        aspect="auto",
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=vmax),
    )
    axes[0].set_xticks(range(len(lift.columns)), lift.columns, rotation=30, ha="right")
    axes[0].set_yticks(range(len(components)), components)
    axes[0].set_xlabel("Behaviour")
    axes[0].set_ylabel("Mixture component")
    axes[0].set_title("Joint responsibility lift")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axes[0].text(column, row, f"{values[row, column]:.2f}", ha="center", va="center", fontsize=7)
    figure.colorbar(image, ax=axes[0], label="Lift relative to behavioural baseline")

    y = np.arange(len(components))
    axes[1].barh(y - .18, prior_support.loc[components, "mean_responsibility"], height=.36, label="mean prior $w_k$")
    axes[1].barh(y + .18, joint_support.loc[components, "mean_responsibility"], height=.36, label="mean posterior $r_k$")
    axes[1].set_ylim(len(components) - .5, -.5)
    axes[1].set_xlabel("Mean responsibility")
    axes[1].set_title("Component utilization")
    axes[1].legend(fontsize=8, loc="lower right")
    axes[1].tick_params(axis="y", left=False, labelleft=False)

    axes[2].barh(y, joint_support.loc[components, "effective_sample_size"], height=.65, color="#4C78A8")
    axes[2].set_ylim(len(components) - .5, -.5)
    axes[2].set_xlabel("Effective sample size")
    axes[2].set_title("Joint posterior support")
    axes[2].tick_params(axis="y", left=False, labelleft=False)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def bootstrap_scanpaths(
    responsibilities: np.ndarray,
    labels: pd.Series,
    scanpath_ids: pd.Series,
    behaviours: list[str],
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Bootstrap statistics by resampling complete reader--text scanpaths.

    Rows within a scanpath are deliberately never sampled independently: their
    temporal dependence is retained by sampling the entire scanpath as one
    unit.  ``lift_samples`` has shape (replicates, components, behaviours).
    """
    one_hot = np.column_stack([(labels == behaviour).to_numpy(float) for behaviour in behaviours])
    group_codes, group_names = pd.factorize(scanpath_ids, sort=True)
    group_indices = [np.flatnonzero(group_codes == group) for group in range(len(group_names))]
    observation_count, component_count = responsibilities.shape
    point_association, point_frequency, point_lift = association_matrix(
        responsibilities, labels, behaviours
    )
    point_support = component_support(responsibilities)

    rng = np.random.default_rng(seed)
    lift_samples = np.empty((replicates, component_count, len(behaviours)))
    association_samples = np.empty_like(lift_samples)
    mean_responsibility_samples = np.empty((replicates, component_count))
    effective_n_samples = np.empty((replicates, component_count))
    for replicate in range(replicates):
        sampled_groups = rng.integers(len(group_indices), size=len(group_indices))
        row_multiplicity = np.zeros(observation_count, dtype=float)
        for group in sampled_groups:
            row_multiplicity[group_indices[group]] += 1.0

        component_mass = row_multiplicity @ responsibilities
        association = responsibilities.T @ (one_hot * row_multiplicity[:, None])
        association /= component_mass[:, None]
        frequency = (row_multiplicity @ one_hot) / row_multiplicity.sum()
        association_samples[replicate] = association
        lift_samples[replicate] = association / frequency[None, :]
        mean_responsibility_samples[replicate] = component_mass / row_multiplicity.sum()
        squared_mass = row_multiplicity @ np.square(responsibilities)
        effective_n_samples[replicate] = np.divide(
            np.square(component_mass), squared_mass,
            out=np.zeros_like(component_mass), where=squared_mass > 0,
        )

    components = point_lift.index
    component_behaviour_rows = []
    for component_index, component in enumerate(components):
        for behaviour_index, behaviour in enumerate(behaviours):
            component_behaviour_rows.append(
                {
                    "component": component,
                    "behaviour": behaviour,
                    "association": point_association.loc[component, behaviour],
                    "lift": point_lift.loc[component, behaviour],
                    "association_ci_lower": np.quantile(association_samples[:, component_index, behaviour_index], .025),
                    "association_ci_upper": np.quantile(association_samples[:, component_index, behaviour_index], .975),
                    "lift_ci_lower": np.quantile(lift_samples[:, component_index, behaviour_index], .025),
                    "lift_ci_upper": np.quantile(lift_samples[:, component_index, behaviour_index], .975),
                    "baseline_frequency": point_frequency.loc[behaviour],
                }
            )
    utilization = pd.DataFrame(
        {
            "mean_joint_responsibility": point_support["mean_responsibility"],
            "mean_joint_responsibility_ci_lower": np.quantile(mean_responsibility_samples, .025, axis=0),
            "mean_joint_responsibility_ci_upper": np.quantile(mean_responsibility_samples, .975, axis=0),
            "joint_effective_sample_size": point_support["effective_sample_size"],
            "joint_effective_sample_size_ci_lower": np.quantile(effective_n_samples, .025, axis=0),
            "joint_effective_sample_size_ci_upper": np.quantile(effective_n_samples, .975, axis=0),
        },
        index=components,
    )
    return pd.DataFrame(component_behaviour_rows), utilization, lift_samples


def save_bootstrap_joint_lift_intervals(intervals: pd.DataFrame, path: Path) -> None:
    """Plot 95% scanpath-bootstrap intervals for every joint-lift cell."""
    behaviours = list(intervals["behaviour"].drop_duplicates())
    components = list(intervals["component"].drop_duplicates())
    figure, axes = plt.subplots(1, len(behaviours), figsize=(18, 9), sharey=True, constrained_layout=True)
    y = np.arange(len(components))
    for axis, behaviour in zip(axes, behaviours):
        subset = intervals[intervals["behaviour"] == behaviour].set_index("component").loc[components]
        estimates = subset["lift"].to_numpy()
        lower = subset["lift_ci_lower"].to_numpy()
        upper = subset["lift_ci_upper"].to_numpy()
        axis.errorbar(
            estimates, y, xerr=np.vstack((estimates - lower, upper - estimates)),
            fmt="o", markersize=4, capsize=2, color="#356E9E", ecolor="#356E9E",
        )
        axis.axvline(1.0, color="black", linestyle="--", linewidth=1)
        axis.set_title(behaviour.replace("_", " "))
        axis.set_xlabel("Joint lift (95% CI)")
        axis.set_xscale("symlog", linthresh=1.0)
        axis.grid(axis="x", alpha=.2)
    axes[0].set_yticks(y, components)
    axes[0].set_ylabel("Mixture component")
    axes[0].invert_yaxis()
    figure.suptitle("Scanpath-bootstrap uncertainty for joint component–behaviour lift", fontsize=13)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def labels_for_session(
    labels: pd.DataFrame, session_metadata: pd.DataFrame
) -> pd.DataFrame:
    """Left-join behavioural labels onto every target in a model sequence."""
    label_columns = [
        "text", "reader", "fixid", "behavioral_label", "is_consecutive_original_fixation",
    ]
    labels = labels[label_columns]
    rows = session_metadata[["text", "reader", "fixid"]].copy()
    rows["_sequence_position"] = np.arange(len(rows))
    rows = rows.merge(labels, how="left", on=["text", "reader", "fixid"], validate="one_to_one")
    return rows.sort_values("_sequence_position", kind="stable").drop(columns="_sequence_position")


def main(args: argparse.Namespace) -> None:
    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        model_name = f"TRANSFORMER_{args.filtering.upper()}_COV_LM"
        checkpoint_path = args.data / "saccade" / model_name / "best_model.pt"
    checkpoint_path = checkpoint_path.resolve()
    output_dir = args.output
    if output_dir is None:
        output_dir = checkpoint_path.parent / "behavioral_responsibility_analysis"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model, is_lm = checkpoint_model(checkpoint_path)
    dataset_type = MecoDatasetLM if is_lm else MecoDataset
    test_dataset = dataset_type(mode="test", filtering=args.filtering, datadir=args.data)
    labels = pd.read_csv(args.labels)
    required_label_columns = {
        "text", "reader", "fixid", "behavioral_label", "is_consecutive_original_fixation",
    }
    missing = required_label_columns.difference(labels.columns)
    if missing:
        raise ValueError(f"Label CSV is missing: {sorted(missing)}")

    event_frames: list[pd.DataFrame] = []
    with torch.no_grad():
        for item_index, item in enumerate(test_dataset.items):
            text, reader = item[:2]
            inputs, positions, saccades = test_dataset[item_index]
            session_metadata = test_dataset.meco_df[
                (test_dataset.meco_df["text"] == text)
                & (test_dataset.meco_df["reader"] == reader)
            ].copy()
            session_labels = labels_for_session(labels, session_metadata)
            if len(session_metadata) != len(positions):
                raise ValueError(
                    f"Metadata/model length mismatch for text={text}, reader={reader}: "
                    f"{len(session_metadata)} rows versus {len(positions)} model targets"
                )

            outputs = model(inputs.unsqueeze(0))
            weights, means, saccade_means, covariances, saccade_scales = outputs
            joint, spatial, priors = log_responsibilities(
                weights, means, saccade_means, covariances, saccade_scales,
                positions.unsqueeze(0), saccades.unsqueeze(0),
            )
            frame = session_labels.reset_index(drop=True)
            joint_np = joint.squeeze(0).cpu().numpy()
            spatial_np = spatial.squeeze(0).cpu().numpy()
            prior_np = priors.squeeze(0).cpu().numpy()
            for component in range(joint_np.shape[1]):
                frame[f"joint_responsibility_{component}"] = joint_np[:, component]
                frame[f"spatial_responsibility_{component}"] = spatial_np[:, component]
                frame[f"prior_weight_{component}"] = prior_np[:, component]
            event_frames.append(frame)

    events = pd.concat(event_frames, ignore_index=True)
    require_consecutive = args.require_consecutive_original_fixation or args.filtering == "raw"
    valid = events["behavioral_label"].isin(PRIMARY_BEHAVIOURS)
    if require_consecutive:
        valid &= events["is_consecutive_original_fixation"].astype(bool)
    analysis_events = events.loc[valid].copy()
    if analysis_events.empty:
        raise ValueError("No events remained after behavioural/continuity filtering.")

    component_count = len([column for column in events if column.startswith("joint_responsibility_")])
    joint_array = analysis_events[
        [f"joint_responsibility_{component}" for component in range(component_count)]
    ].to_numpy()
    spatial_array = analysis_events[
        [f"spatial_responsibility_{component}" for component in range(component_count)]
    ].to_numpy()
    prior_array = analysis_events[
        [f"prior_weight_{component}" for component in range(component_count)]
    ].to_numpy()

    matrices: dict[str, tuple[pd.DataFrame, pd.Series, pd.DataFrame]] = {
        "joint": association_matrix(joint_array, analysis_events["behavioral_label"], PRIMARY_BEHAVIOURS),
        "spatial": association_matrix(spatial_array, analysis_events["behavioral_label"], PRIMARY_BEHAVIOURS),
        "prior": association_matrix(prior_array, analysis_events["behavioral_label"], PRIMARY_BEHAVIOURS),
    }
    for name, (association, frequencies, lift) in matrices.items():
        association.to_csv(output_dir / f"{name}_association.csv")
        lift.to_csv(output_dir / f"{name}_lift.csv")
        frequencies.to_csv(output_dir / f"{name}_baseline_frequency.csv")
        save_heatmap(
            association,
            output_dir / f"{name}_association.png",
            f"{name.capitalize()} component–behaviour association",
            lift=False,
        )
        save_heatmap(
            lift,
            output_dir / f"{name}_lift.png",
            f"{name.capitalize()} component–behaviour lift",
            lift=True,
        )
    supports = {}
    for name, responsibilities in {
        "joint": joint_array,
        "spatial": spatial_array,
        "prior": prior_array,
    }.items():
        supports[name] = component_support(responsibilities)
        supports[name].to_csv(output_dir / f"{name}_component_support.csv")
    prior_support = supports["prior"]
    joint_support = supports["joint"]
    pd.DataFrame(
        {
            "mean_prior_weight": prior_support["mean_responsibility"],
            "mean_joint_responsibility": joint_support["mean_responsibility"],
            "prior_total_mass": prior_support["total_responsibility"],
            "joint_total_mass": joint_support["total_responsibility"],
            "prior_effective_sample_size": prior_support["effective_sample_size"],
            "joint_effective_sample_size": joint_support["effective_sample_size"],
        }
    ).to_csv(output_dir / "component_utilization.csv")
    save_joint_lift_with_utilization(
        matrices["joint"][2],
        prior_support,
        joint_support,
        output_dir / "joint_lift_with_utilization.png",
    )

    if args.bootstrap_replicates:
        scanpath_ids = analysis_events["text"].astype(str) + "::" + analysis_events["reader"].astype(str)
        bootstrap_intervals, bootstrap_utilization, joint_lift_samples = bootstrap_scanpaths(
            joint_array,
            analysis_events["behavioral_label"],
            scanpath_ids,
            PRIMARY_BEHAVIOURS,
            args.bootstrap_replicates,
            args.bootstrap_seed,
        )
        bootstrap_intervals.to_csv(output_dir / "joint_lift_scanpath_bootstrap.csv", index=False)
        bootstrap_utilization.to_csv(output_dir / "joint_utilization_scanpath_bootstrap.csv")
        np.save(output_dir / "joint_lift_scanpath_bootstrap_samples.npy", joint_lift_samples)
        save_bootstrap_joint_lift_intervals(
            bootstrap_intervals,
            output_dir / "joint_lift_scanpath_bootstrap_intervals.png",
        )

    events.to_csv(output_dir / "test_event_responsibilities.csv", index=False)
    summary = {
        "checkpoint": str(checkpoint_path),
        "uses_lm_embeddings": is_lm,
        "total_test_events": int(len(events)),
        "included_events": int(len(analysis_events)),
        "filtering": args.filtering,
        "require_consecutive_original_fixation": require_consecutive,
        "behaviours": PRIMARY_BEHAVIOURS,
        "component_count": component_count,
        "included_behaviour_counts": analysis_events["behavioral_label"].value_counts().to_dict(),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed if args.bootstrap_replicates else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print("\nJoint association lift:")
    print(matrices["joint"][2].round(3).to_string())
    print(f"\nWrote results to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--filtering", choices=["filtered", "raw"], default="filtered")
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Defaults to behavioral_responsibility_analysis beside the selected checkpoint.",
    )
    parser.add_argument(
        "--require-consecutive-original-fixation",
        action="store_true",
        help="Restrict D to transitions consecutive in the original, unfiltered scanpath.",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=1000,
        help="Number of reader--text scanpath bootstrap replicates; set 0 to skip (default: 1000).",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=20261029)
    main(parser.parse_args())
