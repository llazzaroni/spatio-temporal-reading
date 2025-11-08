import pandas as pd

# Project specific imports
from exploration.statistics import saccade_vs_chars, duration_vs_chars, duration_distributions, empirical_regression_probability_vs_chars
from exploration.visualize import plot_durations_vs_chars, plot_saccade_vs_chars, box_plots_duration_distributions, plot_distributions, plot_empirical_regression_prob_vs_chars, plot_empirical_regression_prob_vs_durs, plot_empirical_regression_prob_vs_durs_pre, plot_jump_vs_surprisal_rank

# Read the csvs
meco_df = pd.read_csv("/Users/lorenzolazzaroni/Documents/Programming/Python/Research in DS/data/dataset_cached/hp_augmented_meco_100_1000_1_10.csv")
texts_df = pd.read_csv("/Users/lorenzolazzaroni/Documents/Programming/Python/Research in DS/data/dataset_cached/hp_eng_texts_100_1000_1_10.csv")


# use a function from visualize.py. Example:
plot_saccade_vs_chars(meco_df, texts_df)