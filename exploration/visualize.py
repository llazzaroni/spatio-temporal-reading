import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Project specific imports
from exploration.statistics import saccade_vs_chars, duration_vs_chars, duration_distributions, empirical_regression_probability_vs_chars, empirical_regression_probability_vs_dur, empirical_regression_probability_vs_dur_pre, jump_vs_surprisal_rank

def plot_saccade_vs_chars(meco_df, texts_df):
    saccades, lengths = saccade_vs_chars(meco_df, texts_df)
    #lengths = np.abs(lengths)

    #corr = np.corrcoef(saccades, lengths)[0, 1]
    #print("Pearson r:", corr)
    # Pearson correlation factor:
    # 0.219 excluding the nans from the original table
    # 0.195 without excluding them

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.scatter(saccades, lengths, alpha=0.5, s=8)
    ax.set_xlabel("Saccade duration")
    ax.set_ylabel("Chars spanned")
    ax.set_title("Saccades vs chars")
    ax.set_xlim(0, 0.3)    # e.g., 0–400 ms
    ax.set_ylim(-350, 350) 
    plt.show()

def plot_durations_vs_chars(meco_df, texts_df):
    durations, lengths = duration_vs_chars(meco_df, texts_df)
    #lengths = np.abs(lengths)

    #corr = np.corrcoef(durs, lengths)[0, 1]
    #print("Pearson r:", corr)
    # Pearson correlation factor:
    # -0.0245 excluding the nans from the original table
    # -0.0223 without excluding them

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.scatter(durations, lengths, alpha=0.5, s=8)
    ax.set_xlabel("Gaze duration")
    ax.set_ylabel("Chars spanned")
    ax.set_title("Durations vs chars")
    ax.set_xlim(0, 0.6)    # e.g., 0–400 ms
    ax.set_ylim(-300, 300) 
    plt.show()

def box_plots_duration_distributions(meco_df, texts_df):
    durations_backwards, durations_forward = duration_distributions(meco_df, texts_df)
    plt.boxplot([durations_backwards, durations_forward], labels=["backward", "forward"], notch=True, showmeans=True)
    #plt.boxplot([durations_backwards, durations_forward], labels=["backward", "forward"], showfliers=False, showmeans=True, meanline=True)
    plt.ylabel("Value")
    plt.title("Distributions backward vs forward")
    plt.show()

def plot_distributions(meco_df, texts_df):
    durations_backwards, durations_forward = duration_distributions(meco_df, texts_df)
    plt.hist(durations_backwards, bins='fd', density=True, alpha=0.6, label='backward')  # 'fd' = Freedman–Diaconis
    plt.hist(durations_forward, bins='fd', density=True, alpha=0.6, label='forward')
    plt.xlabel('Value'); plt.ylabel('Density'); plt.legend(); plt.show()

def plot_empirical_regression_prob_vs_chars(meco_df, texts_df):
    probabilities, indices = empirical_regression_probability_vs_chars(meco_df, texts_df)
    norm_indices = []
    for index in indices:
        norm_indices.append((index[0] + index[1])/2)
    x = np.arange(len(probabilities))
    plt.plot(x, probabilities, marker="o")
    plt.xlabel("Characters spanned (bin)")
    plt.ylabel("Regression probability")
    plt.xticks(x[::2], [f"{c:.0f}" for c in norm_indices[::2]], rotation=0)
    plt.tight_layout()
    plt.show()

def plot_empirical_regression_prob_vs_durs(meco_df):
    probabilities, indices = empirical_regression_probability_vs_dur(meco_df)
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
    plt.show()

def plot_empirical_regression_prob_vs_durs_pre(meco_df):
    probabilities, indices = empirical_regression_probability_vs_dur_pre(meco_df)
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
    plt.show()

def plot_jump_vs_surprisal_rank(meco_df, texts_df):
    span, rank = jump_vs_surprisal_rank(meco_df, texts_df)
    x = np.arange(len(span))
    fig, ax = plt.subplots(figsize=(6,4), dpi=150)
    ax.plot(np.arange(len(rank)), rank, marker="o")
    ax.set_xlabel("Characters spanned")
    ax.set_ylabel("Rank of the landing char as of char surprisal")
    step = 6
    ax.set_xticks(np.arange(len(rank))[::step])
    ax.set_xticklabels([f"{c:.1f}" for c in span][::step])
    fig.tight_layout()
    plt.show()


