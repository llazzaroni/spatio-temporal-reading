from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import json
import torch

from spatio_temporal_reading.data.data import MecoDataset

def compute_means_sacc_filtered(args):
    datadir = Path(args.data)

    train_ds = MecoDataset(mode="train", filtering="filtered", datadir=datadir)

    dataloader_kwargs = dict(
        batch_size=1,
        num_workers=1,
        pin_memory=False,
        persistent_workers=False,
    )

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        **dataloader_kwargs,
    )

    lengths = []
    dx_avgs = []
    dy_avgs = []
    dur_avgs = []
    start_avgs = []
    reader_indices = []
    char_level_surp_avgs = []
    word_level_surp_avgs = []
    len_avgs = []
    freq_avgs = []

    for i, item in enumerate(train_loader):
        input_model, positions_target, saccades_target = item

        # We compute means on input_model[:, 1:, ...], so weights must match that length.
        lengths.append(input_model.shape[1] - 1)

        history_points = input_model[:, 1:, :2].squeeze(0)
        history_points_mean = history_points.mean(dim=0)
        dx_avgs.append(history_points_mean[0].item())
        dy_avgs.append(history_points_mean[1].item())

        durations = input_model[:, 1:, 2].squeeze(0).squeeze(-1)
        durations_mean = durations.mean().item()
        dur_avgs.append(durations_mean)

        start = input_model[:, 1:, 3].squeeze(0).squeeze(-1)
        start_mean = start.mean().item()
        start_avgs.append(start_mean)

        reader_idx = input_model[:, 1:, 4:-9].squeeze(0).mean(dim=0)
        reader_indices.append(reader_idx.detach().cpu().numpy())

        char_level_surp = input_model[:, 1:, -9].squeeze(0)
        char_level_surp_mean = char_level_surp.mean().item()
        char_level_surp_avgs.append(char_level_surp_mean)

        word_level_surp = input_model[:, 1:, -8].squeeze(0)
        word_level_surp_mean = word_level_surp.mean().item()
        word_level_surp_avgs.append(word_level_surp_mean)

        token_len = input_model[:, 1:, -7].squeeze(0)
        len_mean = token_len.mean().item()
        len_avgs.append(len_mean)

        freq = input_model[:, 1:, -6].squeeze(0)
        freq_mean = freq.mean().item()
        freq_avgs.append(freq_mean)

    lengths = np.array(lengths, dtype=float)
    dx_avgs = np.array(dx_avgs, dtype=float)
    dy_avgs = np.array(dy_avgs, dtype=float)
    dur_avgs = np.array(dur_avgs, dtype=float)
    start_avgs = np.array(start_avgs, dtype=float)
    reader_indices = np.array(reader_indices, dtype=float)
    char_level_surp_avgs = np.array(char_level_surp_avgs, dtype=float)
    word_level_surp_avgs = np.array(word_level_surp_avgs, dtype=float)
    len_avgs = np.array(len_avgs, dtype=float)
    freq_avgs = np.array(freq_avgs, dtype=float)



    dx_mean = np.average(dx_avgs, weights=lengths)
    dy_mean = np.average(dy_avgs, weights=lengths)
    dur_mean = np.average(dur_avgs, weights=lengths)
    start_mean = np.average(start_avgs, weights=lengths)
    reader_indices_mean = np.average(reader_indices, axis=0, weights=lengths)
    char_level_surp_mean = np.average(char_level_surp_avgs, weights=lengths)
    word_level_surp_mean = np.average(word_level_surp_avgs, weights=lengths)
    len_mean = np.average(len_avgs, weights=lengths)
    freq_mean = np.average(freq_avgs, weights=lengths)


    means = {
        "dx_mean": float(dx_mean),
        "dy_mean": float(dy_mean),
        "dur_mean": float(dur_mean),
        "start_mean": float(start_mean),
        "reader_indices_mean": reader_indices_mean.tolist(),
        "char_level_surp_mean": float(char_level_surp_mean),
        "word_level_surp_mean": float(word_level_surp_mean),
        "len_mean": float(len_mean),
        "freq_mean": float(freq_mean),
    }

    output_path = datadir / "means_sacc_filtered.json"
    with output_path.open("w") as f:
        json.dump(means, f, indent=2)

    return means


def baseline_tensor_sacc(datadir, length_seq, input_model):
    path = datadir / "means_sacc_filtered.json"
    with path.open("r") as f:
        means = json.load(f)
    reader_vec = input_model[0, 0, 4:-9].detach().cpu().tolist()
    baseline = [
        means["dx_mean"], means["dy_mean"], means["dur_mean"], means["start_mean"],
        *reader_vec,
        means["char_level_surp_mean"], means["word_level_surp_mean"],
        means["len_mean"], means["freq_mean"],
        0, 0, 0, 0, 0
    ]
    baseline_1d = torch.tensor(baseline, dtype=torch.float32)
    baseline_td = baseline_1d.unsqueeze(0).expand(length_seq, -1)

    baseline_first = [
        0, 0, 0, 0,
        *reader_vec,
        0, 0, 0, 0, 0, 0, 0, 0, 1             
    ]
    baseline_first = torch.tensor(baseline_first, dtype=torch.float32)

    baseline_complete = torch.cat([baseline_first.unsqueeze(0), baseline_td], dim=0)
    baseline_complete = baseline_complete.unsqueeze(0)
    
    return baseline_complete
