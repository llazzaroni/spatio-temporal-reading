import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from submodule.src.visuals.eval_plots import plot_llr_violins
from submodule.eval.evalu import bootstrap_mean_difference

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
        models = ["RME_CSS_WS_FILTERED", "RME_CSS_WS_RAW", "BASE_LF_RAW", "BASE_SHP_RAW", "CSS_RAW", "poisson_raw_baseline", "RME_CSS_CHAR_LEN_FREQ_RAW", "RME_CSS_CHAR_WORD_LEN_FREQ_RAW", "RME_CSS_CS_RAW", "RME_CSS_DUR_RAW", "RME_CSS_FREQ_RAW", "RME_CSS_LEN_FREQ_RAW", "RME_CSS_LEN_RAW", "RME_CSS_RAW", "RME_CSS_WORD_LEN_FREQ_RAW", "RME_CSS_WS_RAW", "BASE_LF_FILTERED", "BASE_SHP_FILTERED", "CSS_FILTERED", "poisson_filtered_baseline", "RME_CSS_CHAR_LEN_FREQ_FILTERED", "RME_CSS_CHAR_WORD_LEN_FREQ_FILTERED", "RME_CSS_CS_FILTERED", "RME_CSS_DUR_FILTERED", "RME_CSS_FREQ_FILTERED", "RME_CSS_LEN_FREQ_FILTERED", "RME_CSS_LEN_FILTERED", "RME_CSS_FILTERED", "RME_CSS_WORD_LEN_FREQ_FILTERED", "RME_CSS_WS_FILTERED"]
        for model in models:
            model_path = models_path / model
            npy_path = model_path / "negloglikelihoods.npy"
            likelihoods = np.load(npy_path)
            losses[model] = likelihoods

        eff_rme_css_bstrp = bootstrap_mean_difference(
            -losses["RME_CSS_FILTERED"],
            -losses["poisson_filtered_baseline"],
            N=1000,
        )
        eff_css_bstrp = bootstrap_mean_difference(
            -losses["CSS_FILTERED"],
            -losses["poisson_filtered_baseline"],
            N=1000,
        )
        eff_lf_bstrp = bootstrap_mean_difference(
            -losses["BASE_LF_FILTERED"],
            -losses["poisson_filtered_baseline"],
            N=1000,
        )
        eff_hp_bstrp = bootstrap_mean_difference(
            -losses["BASE_SHP_FILTERED"],
            -losses["poisson_filtered_baseline"],
            N=1000,
        )

        eff_rme_css_bstrp_raw = bootstrap_mean_difference(
            -losses["RME_CSS_RAW"], -losses["poisson_raw_baseline"], N=1000, reduce=True
        )
        eff_css_bstrp_raw = bootstrap_mean_difference(
            -losses["CSS_RAW"], -losses["poisson_raw_baseline"], N=1000, reduce=True
        )
        eff_lf_bstrp_raw = bootstrap_mean_difference(
            -losses["BASE_LF_RAW"], -losses["poisson_raw_baseline"], N=1000, reduce=True
        )
        eff_hp_bstrp_raw = bootstrap_mean_difference(
            -losses["BASE_SHP_RAW"], -losses["poisson_raw_baseline"], N=1000, reduce=True
        )

        plot_llr_violins(
            [
                eff_lf_bstrp,
                eff_lf_bstrp_raw,
                eff_hp_bstrp,
                eff_hp_bstrp_raw,
                eff_css_bstrp,
                eff_css_bstrp_raw,
                eff_rme_css_bstrp,
                eff_rme_css_bstrp_raw,
            ],
            labels=[
                "Last \n Fix",
                "Last Fix \n (Raw)",
                "Hawkes",
                "Hawkes \n (Raw)",
                "CSS ",
                "CSS \n (Raw)",
                "RME + CSS",
                "RME + CSS \n (Raw)",
            ],
            title="Bootstrap Estimates of Log-Likelihood Gains on Test Set Across Saccade Models",
            y_label="Delta Log-Likelihood Per Fixation w.r.t. Poisson",
            save_path=Path(__file__).resolve().parent / "saccade_evaluation_wrt_poisson.png",
            fig_size=(9, 6),
        )




    


    


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--compute-stats", default=True, action="store_true")
    p.add_argument("--models-path", type=str)
    p.add_argument("--models", nargs="+")
    p.add_argument("--plot-distributions", default=False, action="store_true")
    p.add_argument("--evaluate-models", default=False, action="store_true")
    args = p.parse_args()
    main(args)


'''
python losses.py --models-path "/Users/lorenzolazzaroni/Documents/Programming/Python/Research in DS/spatio-temporal-reading-proj/data" --models RME_CSS_WS_FILTERED RME_CSS_WS_RAW BASE_LF_RAW BASE_SHP_RAW CSS_RAW poisson_raw_baseline RME_CSS_CHAR_LEN_FREQ_RAW RME_CSS_CHAR_WORD_LEN_FREQ_RAW RME_CSS_CS_RAW RME_CSS_DUR_RAW RME_CSS_FREQ_RAW RME_CSS_LEN_FREQ_RAW RME_CSS_LEN_RAW RME_CSS_RAW RME_CSS_WORD_LEN_FREQ_RAW RME_CSS_WS_RAW BASE_LF_FILTERED BASE_SHP_FILTERED CSS_FILTERED poisson_filtered_baseline RME_CSS_CHAR_LEN_FREQ_FILTERED RME_CSS_CHAR_WORD_LEN_FREQ_FILTERED RME_CSS_CS_FILTERED RME_CSS_DUR_FILTERED RME_CSS_FREQ_FILTERED RME_CSS_LEN_FREQ_FILTERED RME_CSS_LEN_FILTERED RME_CSS_FILTERED RME_CSS_WORD_LEN_FREQ_FILTERED RME_CSS_WS_FILTERED TRANSFORMER_FILTERED TRANSFORMER_RAW
python losses.py --models-path "/Users/lorenzolazzaroni/Documents/Programming/Python/Research in DS/spatio-temporal-reading-proj/data" --models TRANSFORMER_RAW RME_CSS_LEN_FREQ_RAW --plot-distributions
'''