import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Project specific imports
import exploration.statistics as stats

def plot_saccades_vs_chars(meco_df, outputpath, xlim = [0, 0.3], ylim = [-150, 150]):
    # Load from stats
    saccades, lengths = stats.saccades_vs_chars(meco_df)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.scatter(saccades, lengths, alpha=0.5, s=8)
    ax.set_xlabel("Saccade duration")
    ax.set_ylabel("Chars spanned")
    ax.set_title("Saccades vs chars")
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    fig.tight_layout()
    fig.savefig(outputpath / "sacc_vs_char.png")
    plt.close(fig)

def plot_durations_vs_chars(meco_df, outputpath, xlim = [0, 0.7], ylim = [-150, 150]):
    # Load from stats
    durations, lengths = stats.durations_vs_chars(meco_df, False)
    
    # Plot
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.scatter(durations, lengths, alpha=0.5, s=8)
    ax.set_xlabel("Fixation duration")
    ax.set_ylabel("Chars spanned")
    ax.set_title("Durations vs chars")
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    fig.tight_layout()
    fig.savefig(outputpath / "dur_vs_char.png")
    plt.close(fig)

def plot_durations_vs_chars_pre(meco_df, outputpath, xlim = [0, 0.7], ylim = [-150, 150]):
    # Load from stats
    durations, lengths = stats.durations_vs_chars(meco_df, True)
    
    # Plot
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.scatter(durations, lengths, alpha=0.5, s=8)
    ax.set_xlabel("Fixation duration")
    ax.set_ylabel("Chars spanned")
    ax.set_title("Durations vs chars")
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    fig.tight_layout()
    fig.savefig(outputpath / "dur_vs_char_pre.png")
    plt.close(fig)

def box_plots_duration_distributions(meco_df, outputpath):
    durations_backwards, durations_forward = stats.duration_distributions(meco_df)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.boxplot([durations_backwards, durations_forward], labels=["backward", "forward"], notch=True, showmeans=True)
    ax.set_ylabel("Value")
    ax.set_title("Distributions backward vs forward")
    fig.tight_layout()
    fig.savefig(outputpath / "duration_distributions_boxplots.png")
    plt.close(fig)

def plot_distributions(meco_df, outputpath):
    durations_backwards, durations_forward = stats.duration_distributions(meco_df)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.hist(durations_backwards, bins="fd", density=True, alpha=0.6, label="backward")
    ax.hist(durations_forward, bins="fd", density=True, alpha=0.6, label="forward")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outputpath / "duration_distributions_hist.png")
    plt.close(fig)

def plot_empirical_regression_prob_vs_chars(meco_df, outputpath):
    probabilities, indices = stats.empirical_regression_probability_vs_chars(meco_df)
    norm_indices = []
    for index in indices:
        norm_indices.append((index[0] + index[1])/2)
    x = np.arange(len(probabilities))
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.plot(x, probabilities, marker="o")
    ax.set_xlabel("Characters spanned (bin)")
    ax.set_ylabel("Regression probability")
    ax.set_xticks(x[::2])
    ax.set_xticklabels([f"{c:.0f}" for c in norm_indices[::2]], rotation=0)
    fig.tight_layout()
    fig.savefig(outputpath / "reg_prob_vs_char.png")
    plt.close(fig)

def plot_empirical_regression_prob_vs_durs(meco_df, outputpath):
    probabilities, indices = stats.empirical_regression_probability_vs_dur(meco_df)
    norm_indices = []
    for index in indices:
        norm_indices.append((index[0] + index[1])/2)
    x = np.arange(len(probabilities))
    fig, ax = plt.subplots(figsize=(6,4), dpi=150)
    ax.plot(np.arange(len(probabilities)), probabilities, marker="o")
    ax.set_xlabel("Next gaze duration (bin, log scale)")
    ax.set_ylabel("Regression probability")
    step = 6
    ax.set_xticks(np.arange(len(probabilities))[::step])
    ax.set_xticklabels([f"{c:.1f}" for c in norm_indices][::step])
    fig.tight_layout()
    fig.savefig(outputpath / "reg_prob_vs_dur.png")
    plt.close(fig)

def plot_empirical_regression_prob_vs_durs_pre(meco_df, outputpath):
    probabilities, indices = stats.empirical_regression_probability_vs_dur_pre(meco_df)
    norm_indices = []
    for index in indices:
        norm_indices.append((index[0] + index[1])/2)
    x = np.arange(len(probabilities))
    fig, ax = plt.subplots(figsize=(6,4), dpi=150)
    ax.plot(np.arange(len(probabilities)), probabilities, marker="o")
    ax.set_xlabel("Previous gaze duration (bin, log scale)")
    ax.set_ylabel("Regression probability")
    step = 6
    ax.set_xticks(np.arange(len(probabilities))[::step])
    ax.set_xticklabels([f"{c:.1f}" for c in norm_indices][::step])
    fig.tight_layout()
    fig.savefig(outputpath / "reg_prob_vs_dur_pre.png")
    plt.close(fig)

def plot_distributions_next_fixation(meco_df, outputpath):
    forward, backward = stats.saccade_length_forward_vs_backward(meco_df)

    bin_width = 3
    all_data = np.concatenate([forward, backward])
    bins = np.arange(all_data.min(), all_data.max() + bin_width, bin_width)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.hist(forward, bins=bins, density=True, alpha=0.6, label='forward')
    ax.hist(backward, bins=bins, density=True, alpha=0.6, label='backward')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.set_xlim(-150, 150)
    fig.tight_layout()
    fig.savefig(outputpath / "forward_backward_distr.png")
    plt.close(fig)

def plot_jump_vs_surprisal_rank(meco_df, texts_df, outputpath):
    span, rank = stats.jump_vs_surprisal_rank(meco_df, texts_df)
    x = np.arange(len(span))
    fig, ax = plt.subplots(figsize=(6,4), dpi=150)
    ax.plot(np.arange(len(rank)), rank, marker="o")
    ax.set_xlabel("Characters spanned")
    step=3
    ax.set_xticks(np.arange(len(rank))[::step])
    ax.set_xticklabels([f"{c:.1f}" for c in span][::step])
    fig.tight_layout()
    fig.savefig(outputpath / "jump_vs_surprisal_rank.png")
    plt.close(fig)
