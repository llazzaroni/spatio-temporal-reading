from pathlib import Path
from dataclasses import dataclass

from submodule.scripts.model_cards import MODELS
from spatio_temporal_reading import train_baseline

@dataclass
class RunConfig:

    epochs: int = 30
    batch_size: int = 512
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    optimizer: str = "SGDNesterov"  # Adam | SGDNesterov
    gradient_clipping: bool = True
    patience: int = 5
    lr_rescaling: float = 0.99
    training: str = "true"  # "true" | "false"
    final_testing: str = "true"  # "true" | "false"
    splitting_procedure: str = "random_shuffle"
    subset: str = "false"  # "true" | "false"
    subset_size: int = 2_000
    dataset_filtering: str = "raw"  # "filtered" | "raw"
    model_type: str = "saccade"  # "saccade" | "duration"
    missing_value_effects: str = "linear_term"  # "linear_term" | "ignore" |
    saccade_likelihood: str = (
        "ExtendedHawkesProcess"  # "HomogenousPoisson" | "StandardHawkesProcess" | "ExtendedHawkesProcess", "LastFixationModel"
    )
    saccade_predictors_funcs: str = "past_position_reader_char_word_len_freq"
    # "past_position" | "past_position_reader" |
    # "past_position_reader_duration" | "past_position_reader_char" | "past_position_reader_word"
    dur_likelihood: str = (
        "normal"  # "rayleigh" | "exponential" | "lognormal" | "normal"
    )
    duration_predictors_funcs: str = "dur_model_reader_dur_conv_features"
    # dur_model_baseline
    # dur_model_reader_char_conv_features, dur_model_reader_dur_conv_features, dur_model_reader_word_conv_features
    load_checkpoint: str = "false"
    checkpoint_path: Path | None = None
    strict_load: bool = False
    # reproducibility / hardware
    seed: int = 124
    nworkers: int = 0
    # directory for the experiments
    experiment_dir: str = str("runs")
    # directory for the specific run
    directory_name: str = f"hp_{saccade_likelihood}"
    # we set this to None whenever we want to test on the same model we are training
    # if we set testing = True, training = False, we will not train a model, so we can set this to the directory of the model we want to test
    test_model_dir: Path | None = (
        "/Users/francescoignaziore/Projects/fine-grained-model-reading-behaviour/cluster_runs/saccade/rme_css_len_raw_2025-06-06_05-12-54-487/best_model"
    )
    # In the Meco dataset durations and saccades are expressed in milliseconds
    # interarrival saccades times between two consequent saccades have a median of 27 ms.
    # in an exponential kernel, a * exp-b(27) is an extremely small value unless there is a big value of (a,b) to counterbalance it.
    # In order to avoid numerical issues, we divide the saccade-intervals by 1000 to convert them to seconds, to allow for a range of values of plausible candidate (a,b) that is more stable for optimization.
    # scaling factors to avoid numerical issues
    division_factor_space: int = 100
    division_factor_time: int = 1000
    division_factor_durations: int = 1
    past_timesteps_duration_baseline_k: int = 10
    # initialization of parameters for convolution gamma kernel
    alpha_g: float = 0.1
    delta_g: float = 0.1
    beta_g: float = 0.1

# Handle the paths to load from
MODELS["poisson_raw"]["load_checkpoint"] = False
MODELS["poisson_raw"]["load_path"] = None
MODELS["last_fix_raw"]["load_checkpoint"] = False
MODELS["last_fix_raw"]["load_path"] = None
MODELS["stand_hawkes_raw"]["load_checkpoint"] = False
MODELS["stand_hawkes_raw"]["load_path"] = None
MODELS["css_raw"]["load_checkpoint"] = False
MODELS["css_raw"]["load_path"] = None
MODELS["rme_css_raw"]["load_checkpoint"] = False
MODELS["rme_css_raw"]["load_path"] = None
MODELS["rme_css_cs_raw"]["load_checkpoint"] = True
MODELS["rme_css_cs_raw"]["load_path"] = "RME_CSS_RAW"
MODELS["rme_css_ws_raw"]["load_checkpoint"] = True
MODELS["rme_css_ws_raw"]["load_path"] = "RME_CSS_RAW"
MODELS["rme_css_dur_raw"]["load_checkpoint"] = True
MODELS["rme_css_dur_raw"]["load_path"] = "RME_CSS_RAW"
MODELS["rme_css_len_raw"]["load_checkpoint"] = True
MODELS["rme_css_len_raw"]["load_path"] = "RME_CSS_RAW"
MODELS["rme_css_freq_raw"]["load_checkpoint"] = True
MODELS["rme_css_freq_raw"]["load_path"] = "RME_CSS_RAW"
MODELS["rme_css_len_freq_raw"]["load_checkpoint"] = True
MODELS["rme_css_len_freq_raw"]["load_path"] = "RME_CSS_RAW"
MODELS["rme_css_char_len_freq_raw"]["load_checkpoint"] = True
MODELS["rme_css_char_len_freq_raw"]["load_path"] = "RME_CSS_RAW"
MODELS["rme_css_word_len_freq_raw"]["load_checkpoint"] = True
MODELS["rme_css_word_len_freq_raw"]["load_path"] = "RME_CSS_RAW"
MODELS["rme_css_char_word_len_freq_raw"]["load_checkpoint"] = True
MODELS["rme_css_char_word_len_freq_raw"]["load_path"] = "RME_CSS_RAW"

MODELS["poisson_filtered"]["load_checkpoint"] = False
MODELS["poisson_filtered"]["load_path"] = None
MODELS["last_fix_filtered"]["load_checkpoint"] = False
MODELS["last_fix_filtered"]["load_path"] = None
MODELS["stand_hawkes_filtered"]["load_checkpoint"] = False
MODELS["stand_hawkes_filtered"]["load_path"] = None
MODELS["css_filtered"]["load_checkpoint"] = False
MODELS["css_filtered"]["load_path"] = None
MODELS["rme_css_filtered"]["load_checkpoint"] = False
MODELS["rme_css_filtered"]["load_path"] = None
MODELS["rme_css_cs_filtered"]["load_checkpoint"] = True
MODELS["rme_css_cs_filtered"]["load_path"] = "RME_CSS_FILTERED"
MODELS["rme_css_ws_filtered"]["load_checkpoint"] = True
MODELS["rme_css_ws_filtered"]["load_path"] = "RME_CSS_FILTERED"
MODELS["rme_css_dur_filtered"]["load_checkpoint"] = True
MODELS["rme_css_dur_filtered"]["load_path"] = "RME_CSS_FILTERED"
MODELS["rme_css_len_filtered"]["load_checkpoint"] = True
MODELS["rme_css_len_filtered"]["load_path"] = "RME_CSS_FILTERED"
MODELS["rme_css_freq_filtered"]["load_checkpoint"] = True
MODELS["rme_css_freq_filtered"]["load_path"] = "RME_CSS_FILTERED"
MODELS["rme_css_len_freq_filtered"]["load_checkpoint"] = True
MODELS["rme_css_len_freq_filtered"]["load_path"] = "RME_CSS_FILTERED"
MODELS["rme_css_char_len_freq_filtered"]["load_checkpoint"] = True
MODELS["rme_css_char_len_freq_filtered"]["load_path"] = "RME_CSS_FILTERED"
MODELS["rme_css_word_len_freq_filtered"]["load_checkpoint"] = True
MODELS["rme_css_word_len_freq_filtered"]["load_path"] = "RME_CSS_FILTERED"
MODELS["rme_css_char_word_len_freq_filtered"]["load_checkpoint"] = True
MODELS["rme_css_char_word_len_freq_filtered"]["load_path"] = "RME_CSS_FILTERED"

MODELS["dur_baseline_raw"]["load_checkpoint"] = False
MODELS["dur_baseline_raw"]["load_path"] = None
MODELS["dur_rme_raw"]["load_checkpoint"] = False
MODELS["dur_rme_raw"]["load_path"] = None
MODELS["dur_rme_dur_raw"]["load_checkpoint"] = True
MODELS["dur_rme_dur_raw"]["load_path"] = "RME_LN_RAW"
MODELS["dur_rme_cs_raw"]["load_checkpoint"] = True
MODELS["dur_rme_cs_raw"]["load_path"] = "RME_LN_RAW"
MODELS["dur_rme_ws_raw"]["load_checkpoint"] = True
MODELS["dur_rme_ws_raw"]["load_path"] = "RME_LN_RAW"
MODELS["dur_rme_freq_raw"]["load_checkpoint"] = True
MODELS["dur_rme_freq_raw"]["load_path"] = "RME_LN_RAW"
MODELS["dur_rme_len_raw"]["load_checkpoint"] = True
MODELS["dur_rme_len_raw"]["load_path"] = "RME_LN_RAW"
MODELS["dur_rme_len_freq_raw"]["load_checkpoint"] = True
MODELS["dur_rme_len_freq_raw"]["load_path"] = "RME_LN_RAW"
MODELS["dur_rme_char_len_freq_raw"]["load_checkpoint"] = True
MODELS["dur_rme_char_len_freq_raw"]["load_path"] = "RME_LN_RAW"
MODELS["dur_rme_word_len_freq_raw"]["load_checkpoint"] = True
MODELS["dur_rme_word_len_freq_raw"]["load_path"] = "RME_LN_RAW"
MODELS["dur_rme_char_word_len_freq_raw"]["load_checkpoint"] = True
MODELS["dur_rme_char_word_len_freq_raw"]["load_path"] = "RME_LN_RAW"

MODELS["dur_baseline_filtered"]["load_checkpoint"] = False
MODELS["dur_baseline_filtered"]["load_path"] = None
MODELS["dur_rme_filtered"]["load_checkpoint"] = False
MODELS["dur_rme_filtered"]["load_path"] = None
MODELS["dur_rme_dur_filtered"]["load_checkpoint"] = True
MODELS["dur_rme_dur_filtered"]["load_path"] = "RME_LN_FILTERED"
MODELS["dur_rme_cs_filtered"]["load_checkpoint"] = True
MODELS["dur_rme_cs_filtered"]["load_path"] = "RME_LN_FILTERED"
MODELS["dur_rme_ws_filtered"]["load_checkpoint"] = True
MODELS["dur_rme_ws_filtered"]["load_path"] = "RME_LN_FILTERED"
MODELS["dur_rme_freq_filtered"]["load_checkpoint"] = True
MODELS["dur_rme_freq_filtered"]["load_path"] = "RME_LN_FILTERED"
MODELS["dur_rme_len_filtered"]["load_checkpoint"] = True
MODELS["dur_rme_len_filtered"]["load_path"] = "RME_LN_FILTERED"
MODELS["dur_rme_len_freq_filtered"]["load_checkpoint"] = True
MODELS["dur_rme_len_freq_filtered"]["load_path"] = "RME_LN_FILTERED"
MODELS["dur_rme_char_len_freq_filtered"]["load_checkpoint"] = True
MODELS["dur_rme_char_len_freq_filtered"]["load_path"] = "RME_LN_FILTERED"
MODELS["dur_rme_word_len_freq_filtered"]["load_checkpoint"] = True
MODELS["dur_rme_word_len_freq_filtered"]["load_path"] = "RME_LN_FILTERED"
MODELS["dur_rme_char_word_len_freq_filtered"]["load_checkpoint"] = True
MODELS["dur_rme_char_word_len_freq_filtered"]["load_path"] = "RME_LN_FILTERED"



def main(datapath):
    # Hyperparameters
    batch_sizes = [64, 128, 256]
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    weight_decays = [0.0, 1e-4]
    alphas_betas = [(2, 3), (3, 4), (3, 6)]
    delta = 0.5

    cfg = RunConfig()

    for model in MODELS.values():
        ############### skip models that have already been trained. TO REMOVE LATER! ###############
        #if model["directory_name"] == "BASE_LF_RAW" or model["directory_name"] == "BASE_SHP_RAW" or model["directory_name"] == "CSS_RAW" or model["directory_name"] == "poisson_raw_baseline" or model["directory_name"] == "RME_CSS_CHAR_LEN_FREQ_RAW" or model["directory_name"] == "RME_CSS_CS_RAW" or model["directory_name"] == "RME_CSS_DUR_RAW" or model["directory_name"] == "RME_CSS_DUR_RAW" or model["directory_name"] == "RME_CSS_FREQ_RAW" or model["directory_name"] == "RME_CSS_LEN_FREQ_RAW" or model["directory_name"] == "RME_CSS_LEN_RAW" or model["directory_name"] == "RME_CSS_WORD_LEN_FREQ_RAW" or model["directory_name"] == "RME_CSS_WS_RAW":
        #    continue
        ############### skip models that have already been trained. TO REMOVE LATER! ###############
        # Build the cfg
        cfg.model_type = model["model_type"]
        if "saccade_likelihood" in model:
            cfg.saccade_likelihood = model["saccade_likelihood"]
        cfg.dataset_filtering = model["dataset_filtering"]
        if "missing_value_effects" in model:
            cfg.missing_value_effects = model["missing_value_effects"]
        if "saccade_predictors_funcs" in model:
            cfg.saccade_predictors_funcs = model["saccade_predictors_funcs"]
        cfg.directory_name = model["directory_name"]
        if "dur_likelihood" in model:
            cfg.dur_likelihood = model["dur_likelihood"]
        if "duration_predictors_funcs" in model:
            cfg.duration_predictors_funcs = model["duration_predictors_funcs"]

        # Where to load the parameters from
        load_model = model["load_checkpoint"]
        load_path = model["load_path"]
        
        for batch_size in batch_sizes:

            ########################## REMOVE WHEN TRAINING ALSO DURATIONS ##########################
            if model["model_type"] == "duration":
                continue
            ########################## REMOVE WHEN TRAINING ALSO DURATIONS ##########################

            if model["model_type"] == "duration" and batch_size != 128:
                continue
            cfg.batch_size = batch_size
            for learning_rate in learning_rates:
                if model["model_type"] == "duration" and learning_rate == 0.1:
                    continue
                if model["model_type"] == "saccade" and learning_rate == 0.0001:
                    continue
                cfg.learning_rate = learning_rate
                for weight_decay in weight_decays:
                    cfg.weight_decay = weight_decay

                    print("Training a new model. Saving outputs to:", model["directory_name"])

                    if model["model_type"] == "duration":
                        for alpha_beta in alphas_betas:
                            cfg.alpha_g = alpha_beta[0]
                            cfg.beta_g = alpha_beta[1]
                            cfg.delta_g = delta
                            train_baseline.main(datapath, cfg, load_model, load_path)
                    else:
                        train_baseline.main(datapath, cfg, load_model, load_path)

                    
                        