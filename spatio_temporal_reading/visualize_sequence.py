import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch

from spatio_temporal_reading.data.data import MecoDataset
from spatio_temporal_reading.model.model import SimpleModel


def load_model(checkpoint_path):
    # Explicitly allow full checkpoint loading (needed for older saves)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = SimpleModel(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def prepare_single_sample(sample):
    
    positions, durations, starting_times, saccades, reader_emb, features = sample

    # Add batch dimension
    positions = positions.unsqueeze(0)
    durations = durations.unsqueeze(0)
    starting_times = starting_times.unsqueeze(0)
    saccades = saccades.unsqueeze(0)
    reader_emb = reader_emb.unsqueeze(0)
    features = features.unsqueeze(0)

    input_model = torch.cat(
        [
            positions[:, 1:-1, :],
            durations[:, 1:-1].unsqueeze(-1),
            starting_times[:, 1:-1].unsqueeze(-1),
            reader_emb[:, 1:-1, :],
            features[:, 1:-1, :],
        ],
        dim=-1,
    )

    positions_target = positions[:, 2:, :]
    saccades_target = saccades[:, 2:]
    return input_model, positions_target, saccades_target


def collect_predictions(model, sample):
    input_model, positions_target, _ = prepare_single_sample(sample)

    with torch.no_grad():
        weights, positions_model, _ = model(input_model)

    return weights[0], positions_model[0], positions_target[0]  # strip batch dim


def collect_predictions_saccade(model, sample, std_sacc):
    input_model, _, saccades_target = prepare_single_sample(sample)

    with torch.no_grad():
        weights, _, saccades_model = model(input_model)

    # Convert log-normal parameter mu to mean duration: exp(mu + sigma^2 / 2)
    mean_pred = torch.exp(saccades_model + 0.5 * std_sacc**2)

    return weights[0], mean_pred[0], saccades_target[0]


def make_animation_pos(weights, pred_means, targets, output_path, fps: int = 2):

    T, K, _ = pred_means.shape
    w_np = weights.numpy()
    pred_np = pred_means.numpy()
    target_np = targets.numpy()

    # Determine axis limits once to keep them fixed across frames
    all_x = pred_np[:, :, 0].flatten().tolist() + target_np[:, 0].tolist()
    all_y = pred_np[:, :, 1].flatten().tolist() + target_np[:, 1].tolist()
    margin_x = (max(all_x) - min(all_x)) * 0.1 + 1e-3
    margin_y = (max(all_y) - min(all_y)) * 0.1 + 1e-3
    xlim = (min(all_x) - margin_x, max(all_x) + margin_x)
    ylim = (min(all_y) - margin_y, max(all_y) + margin_y)

    fig, ax = plt.subplots(figsize=(6, 6))

    def update(frame_idx):
        ax.clear()
        # Normalize weights per frame to [0, 1] for alpha intensity
        frame_weights = w_np[frame_idx]
        if frame_weights.max() > 0:
            alphas = frame_weights / frame_weights.max()
        else:
            alphas = frame_weights
        colors = [(1, 0, 0, float(a)) for a in alphas]

        ax.scatter(pred_np[frame_idx, :, 0], pred_np[frame_idx, :, 1], c=colors, label="Predicted means (alpha=weight)")
        ax.scatter(target_np[frame_idx, 0], target_np[frame_idx, 1], c="blue", label="Actual position")
        ax.set_title(f"Step {frame_idx + 1}")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.legend(loc="upper right")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    anim = animation.FuncAnimation(fig, update, frames=T, interval=1000 / fps, repeat=False)
    writer = animation.PillowWriter(fps=fps)
    anim.save(str(output_path), writer=writer)
    plt.close(fig)


def make_animation_sacc(weights, pred_means, targets, output_path, fps: int = 2):

    T, K = pred_means.shape
    w_np = weights.numpy()
    pred_np = pred_means.numpy()
    target_np = targets.numpy()

    all_vals = pred_np.flatten().tolist() + target_np.flatten().tolist()
    val_min, val_max = min(all_vals), max(all_vals)
    margin = (val_max - val_min) * 0.1 + 1e-3
    xlim = (val_min - margin, val_max + margin)

    fig, ax = plt.subplots(figsize=(8, 2.5))

    def update(frame_idx):
        ax.clear()
        frame_weights = w_np[frame_idx]
        if frame_weights.max() > 0:
            alphas = frame_weights / frame_weights.max()
        else:
            alphas = frame_weights
        colors = [(1, 0, 0, float(a)) for a in alphas]

        y_pred = np.zeros(K)
        ax.scatter(pred_np[frame_idx], y_pred, c=colors, label="Predicted saccade mean (alpha=weight)")
        ax.scatter(target_np[frame_idx], 0, c="blue", label="Actual saccade")

        ax.set_title(f"Saccade - step {frame_idx + 1}")
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.5, 0.5)
        ax.legend(loc="upper right")
        ax.set_xlabel("Saccade")
        ax.set_yticks([])

    anim = animation.FuncAnimation(fig, update, frames=T, interval=1000 / fps, repeat=False)
    writer = animation.PillowWriter(fps=fps)
    anim.save(str(output_path), writer=writer)
    plt.close(fig)


def load_std_sacc(datapath: Path) -> torch.Tensor:
    with open(datapath / "variances.json", "r") as f:
        variances = json.load(f)
    var_sacc = variances["var_sacc"]
    return torch.tensor(np.sqrt(var_sacc), dtype=torch.float32)

def main(datapath, checkpoint_path, outputpath, args):

    train_ds = MecoDataset(mode="train", datadir=datapath)
    val_ds = MecoDataset(mode="valid", datadir=datapath)

    model = load_model(checkpoint_path)
    std_sacc = load_std_sacc(datapath)

    # Train sample
    train_sample = train_ds[args.train_index]
    train_weights, train_pred_means, train_targets = collect_predictions(model, train_sample)
    make_animation_pos(train_weights, train_pred_means, train_targets, outputpath / "train_pos.gif")
    print(f"Saved train animation to {outputpath}")
    train_w_s, train_pred_s, train_t_s = collect_predictions_saccade(model, train_sample, std_sacc)
    make_animation_sacc(train_w_s, train_pred_s, train_t_s, outputpath / "train_sacc.gif")
    print(f"Saved train saccade animation to {outputpath}")

    # Validation sample
    val_sample = val_ds[args.val_index]
    val_weights, val_pred_means, val_targets = collect_predictions(model, val_sample)
    make_animation_pos(val_weights, val_pred_means, val_targets, outputpath / "val_pos.gif")
    print(f"Saved validation animation to {outputpath}")
    val_w_s, val_pred_s, val_t_s = collect_predictions_saccade(model, val_sample, std_sacc)
    make_animation_sacc(val_w_s, val_pred_s, val_t_s, outputpath / "val_sacc.gif")
    print(f"Saved validation saccade animation to {outputpath}")
