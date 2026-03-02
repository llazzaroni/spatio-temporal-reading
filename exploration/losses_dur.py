import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from submodule.src.visuals.eval_plots import plot_llr_violins
from submodule.eval.evalu import bootstrap_mean_difference
import time

def main(args):
    models_path = Path(args.models_path)

    losses = []
    labels = []

    if args.compute_stats:
        if args.models:
            for model in args.models:
                model = str(model)
                model_path = models_path / model
                npy_path = model_path / "negloglikelihoods.npy"
                likelihoods = np.load(npy_path)
                print(model + ":", likelihoods.mean())
                losses.append(likelihoods)
                labels.append(model)

    if args.plot_distributions:
        xmin = -4
        xmax = 10
        bin_width = 0.1
        bins = np.arange(xmin, xmax + bin_width, bin_width)
        #all_vals = np.concatenate([a.ravel() for a in losses])

        for a, label in zip(losses, labels):
            values = a.ravel()
            values = values[np.isfinite(values)]   # optional: remove NaN/inf
            plt.hist(values, bins=bins, density=True, alpha=0.4, label=label)

        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()
        plt.title("Distributions")
        plt.show()
    
    if args.evaluate_models:
        losses = {}
        models = ["dur_baseline_raw", "duration_baseline_filtered", "RME_LN_CHAR_LEN_FREQ_FILTERED", "RME_LN_CHAR_LEN_FREQ_RAW", "RME_LN_CHAR_WORD_LEN_FREQ_FILTERED", "RME_LN_CHAR_WORD_LEN_FREQ_RAW", "RME_LN_CS_FILTERED", "RME_LN_CS_RAW", "RME_LN_DUR_FILTERED", "RME_LN_DUR_RAW", "RME_LN_FILTERED", "RME_LN_FREQ_FILTERED", "RME_LN_FREQ_RAW", "RME_LN_LEN_FILTERED", "RME_LN_LEN_FREQ_FILTERED", "RME_LN_LEN_FREQ_RAW", "RME_LN_LEN_RAW", "RME_LN_RAW", "RME_LN_WORD_LEN_FREQ_FILTERED", "RME_LN_WORD_LEN_FREQ_RAW", "RME_LN_WS_FILTERED", "RME_LN_WS_RAW"]
        for model in models:
            model_path = models_path / model
            npy_path = model_path / "negloglikelihoods.npy"
            likelihoods = np.load(npy_path)
            losses[model] = likelihoods

        eff_rme = bootstrap_mean_difference(
            -losses["RME_LN_FILTERED"],
            -losses["duration_baseline_filtered"],
            N=1000,
        )

        eff_rme_raw = bootstrap_mean_difference(
            -losses["RME_LN_RAW"],
            -losses["dur_baseline_raw"],
            N=1000,
        )

        eff_rme_len = bootstrap_mean_difference(
            -losses["RME_LN_LEN_FILTERED"],
            -losses["duration_baseline_filtered"],
            N=1000,
        )

        eff_rme_len_raw = bootstrap_mean_difference(
            -losses["RME_LN_LEN_RAW"],
            -losses["dur_baseline_raw"],
            N=1000,
        )

        eff_rme_freq = bootstrap_mean_difference(
            -losses["RME_LN_FREQ_FILTERED"],
            -losses["duration_baseline_filtered"],
            N=1000,
        )

        eff_rme_freq_raw = bootstrap_mean_difference(
            -losses["RME_LN_FREQ_RAW"],
            -losses["dur_baseline_raw"],
            N=1000,
        )

        eff_rme_ws = bootstrap_mean_difference(
            -losses["RME_LN_WS_FILTERED"],
            -losses["duration_baseline_filtered"],
            N=1000,
        )

        eff_rme_ws_raw = bootstrap_mean_difference(
            -losses["RME_LN_WS_RAW"],
            -losses["dur_baseline_raw"],
            N=1000,
        )

        eff_rme_cs = bootstrap_mean_difference(
            -losses["RME_LN_CS_FILTERED"],
            -losses["duration_baseline_filtered"],
            N=1000,
        )

        eff_rme_cs_raw = bootstrap_mean_difference(
            -losses["RME_LN_CS_RAW"],
            -losses["dur_baseline_raw"],
            N=1000,
        )

        eff_rme_dur = bootstrap_mean_difference(
            -losses["RME_LN_DUR_FILTERED"],
            -losses["duration_baseline_filtered"],
            N=1000,
        )

        eff_rme_dur_raw = bootstrap_mean_difference(
            -losses["RME_LN_DUR_RAW"],
            -losses["dur_baseline_raw"],
            N=1000,
        )

        eff_rme_len_freq = bootstrap_mean_difference(
            -losses["RME_LN_LEN_FREQ_FILTERED"],
            -losses["duration_baseline_filtered"],
            N=1000,
        )

        eff_rme_len_freq_raw = bootstrap_mean_difference(
            -losses["RME_LN_LEN_FREQ_RAW"],
            -losses["dur_baseline_raw"],
            N=1000,
        )

        eff_rme_word_len_freq = bootstrap_mean_difference(
            -losses["RME_LN_WORD_LEN_FREQ_FILTERED"],
            -losses["duration_baseline_filtered"],
            N=1000,
        )

        eff_rme_word_len_freq_raw = bootstrap_mean_difference(
            -losses["RME_LN_WORD_LEN_FREQ_RAW"],
            -losses["dur_baseline_raw"],
            N=1000,
        )

        eff_rme_word_len_freq = bootstrap_mean_difference(
            -losses["RME_LN_WORD_LEN_FREQ_FILTERED"],
            -losses["duration_baseline_filtered"],
            N=1000,
        )

        eff_rme_word_len_freq_raw = bootstrap_mean_difference(
            -losses["RME_LN_WORD_LEN_FREQ_RAW"],
            -losses["dur_baseline_raw"],
            N=1000,
        )

        eff_rme_char_len_freq = bootstrap_mean_difference(
            -losses["RME_LN_CHAR_LEN_FREQ_FILTERED"],
            -losses["duration_baseline_filtered"],
            N=1000,
        )

        eff_rme_char_len_freq_raw = bootstrap_mean_difference(
            -losses["RME_LN_CHAR_LEN_FREQ_RAW"],
            -losses["dur_baseline_raw"],
            N=1000,
        )

        eff_rme_char_word_len_freq = bootstrap_mean_difference(
            -losses["RME_LN_CHAR_WORD_LEN_FREQ_FILTERED"],
            -losses["duration_baseline_filtered"],
            N=1000,
        )

        eff_rme_char_word_len_freq_raw = bootstrap_mean_difference(
            -losses["RME_LN_CHAR_WORD_LEN_FREQ_RAW"],
            -losses["dur_baseline_raw"],
            N=1000,
        )

        plot_llr_violins(
            [
                eff_rme,
                eff_rme_raw,
                eff_rme_len,
                eff_rme_len_raw,
                eff_rme_freq,
                eff_rme_freq_raw,
                eff_rme_ws,
                eff_rme_ws_raw,
                eff_rme_cs,
                eff_rme_cs_raw,
                eff_rme_dur,
                eff_rme_dur_raw
            ],
            labels=[
                "rme",
                "rme (Raw)",
                "rme len",
                "rme len (Raw)",
                "rme freq",
                "rme freq (Raw)",
                "rme ws",
                "rme ws (Raw)",
                "rme cs",
                "rme cs (Raw)",
                "rme dur",
                "rme dur (Raw)"
            ],
            title="Bootstrap Estimates of Log-Likelihood Gains on Test Set",
            y_label="Delta Log-Likelihood Per Fixation w.r.t. poisson",
            save_path=Path(__file__).resolve().parent / "duration_evaluation_wrt_poisson.png",
            fig_size=(10, 6),
            step=0.05,
            dpi=1200,
        )
        '''
        transformer_raw_head = bootstrap_mean_difference(
            -losses["TRANSFORMER_RAW_HEADS"], -losses["TRANSFORMER_RAW"], N=1000, reduce=True
        )
        transformer_filtered_head = bootstrap_mean_difference(
            -losses["TRANSFORMER_FILTERED_HEADS"], -losses["TRANSFORMER_FILTERED"], N=1000, reduce=True
        )
        transformer_filtered_lm = bootstrap_mean_difference(
            -losses["TRANSFORMER_FILTERED_LM"], -losses["TRANSFORMER_FILTERED"], N=1000, reduce=True
        )
        transformer_raw_lm = bootstrap_mean_difference(
            -losses["TRANSFORMER_RAW_LM"], -losses["TRANSFORMER_RAW"], N=1000, reduce=True
        )

        plot_llr_violins(
            [
                transformer_filtered_head,
                transformer_raw_head,
                transformer_filtered_lm,
                transformer_raw_lm
            ],
            labels=[
                "Transf Heads",
                "Transf Heads \n (Raw)",
                "Transf LM",
                "Transf LM \n (Raw)"
            ],
            title="Bootstrap Estimates of Log-Likelihood Gains on Test Set",
            y_label="Delta Log-Likelihood Per Fixation w.r.t. transformer baseline",
            save_path=Path(__file__).resolve().parent / "saccade_evaluation_wrt_transformer_complete.png",
            fig_size=(10, 6),
            step=0.25,
            dpi=1200,
        )
        
        transformer_filtered_cov = bootstrap_mean_difference(
            -losses["TRANSFORMER_FILTERED_COV"], -losses["TRANSFORMER_FILTERED"], N=1000, reduce=True
        )
        transformer_raw_cov = bootstrap_mean_difference(
            -losses["TRANSFORMER_RAW_COV"], -losses["TRANSFORMER_RAW"], N=1000, reduce=True
        )
        transformer_filtered_cov_lm = bootstrap_mean_difference(
            -losses["TRANSFORMER_FILTERED_COV_LM"], -losses["TRANSFORMER_FILTERED"], N=1000, reduce=True
        )
        transformer_raw_cov_lm = bootstrap_mean_difference(
            -losses["TRANSFORMER_RAW_COV_LM"], -losses["TRANSFORMER_RAW"], N=1000, reduce=True
        )

        plot_llr_violins(
            [
                transformer_filtered_cov,
                transformer_raw_cov,
                transformer_filtered_cov_lm,
                transformer_raw_cov_lm
            ],
            labels=[
                "Transf Cov",
                "Transf Cov \n (Raw)",
                "Transf LM Cov",
                "Transf LM Cov \n (Raw)"
            ],
            title="Bootstrap Estimates of Log-Likelihood Gains on Test Set",
            y_label="Delta Log-Likelihood Per Fixation w.r.t. transformer baseline",
            save_path=Path(__file__).resolve().parent / "saccade_evaluation_wrt_transformer_cov.png",
            fig_size=(10, 6),
            step=0.5,
            dpi=1200,
        )
        '''

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--compute-stats", default=True, action="store_true")
    p.add_argument("--models-path", type=str)
    p.add_argument("--models", nargs="+")
    p.add_argument("--plot-distributions", default=False, action="store_true")
    p.add_argument("--evaluate-models", default=False, action="store_true")
    args = p.parse_args()
    main(args)
