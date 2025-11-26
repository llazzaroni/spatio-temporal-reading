import pandas as pd
from pathlib import Path

# Project specific imports
import exploration.statistics as statistics
import exploration.visualize as visualize

def make_plots(datapath, outputpath):
    meco_df = pd.read_csv(datapath / "hp_augmented_meco_100_1000_1_10_sacc_idx.csv")
    texts_df = pd.read_csv(datapath / "hp_eng_texts_100_1000_1_10.csv")

    visualize.plot_saccades_vs_chars(meco_df, outputpath)
    visualize.plot_durations_vs_chars(meco_df, outputpath)
    visualize.plot_durations_vs_chars_pre(meco_df, outputpath)
    visualize.box_plots_duration_distributions(meco_df, outputpath)
    visualize.plot_distributions(meco_df, outputpath)
    visualize.plot_empirical_regression_prob_vs_chars(meco_df, outputpath)
    visualize.plot_empirical_regression_prob_vs_durs(meco_df, outputpath)
    visualize.plot_empirical_regression_prob_vs_durs_pre(meco_df, outputpath)
    visualize.plot_distributions_next_fixation(meco_df, outputpath)
    visualize.plot_jump_vs_surprisal_rank(meco_df, texts_df, outputpath)