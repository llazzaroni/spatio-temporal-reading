from torch.utils.data import DataLoader, Subset
import torch
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Any, Dict

from submodule.src.dataset import feature_funcs
from spatio_temporal_reading.submodule_wrappers.dataset import MecoDatasetW
from submodule.src.dataset.utils import collate_fn
from submodule.src.model.neural import MarkedPointProcess
from submodule.src.model.saccades.log_likelihood import set_saccadesNLL
from submodule.src.model.durations.log_likelihood import DurationNLL
from spatio_temporal_reading.submodule_wrappers.trainer import TrainerW


@dataclass
class RunConfig:

    epochs: int = 2
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
    subset: str = "true"  # "true" | "false"
    subset_size: int = 2_000
    dataset_filtering: str = "filtered"  # "filtered" | "raw"
    model_type: str = "saccade"  # "saccade" | "duration"
    missing_value_effects: str = "linear_term"  # "linear_term" | "ignore" |
    saccade_likelihood: str = (
        "HomogenousPoisson"  # "HomogenousPoisson" | "StandardHawkesProcess" | "ExtendedHawkesProcess", "LastFixationModel"
    )
    saccade_predictors_funcs: str = "past_position"
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

class DummyLogger:
    def info(self, *args, **kwargs):
        pass


def main(datapath, outputpath, args):

    cfg = RunConfig
    device = "cpu"
    logger = DummyLogger()

    dataset_kwargs: Dict[str, Any] = dict(
        splitting_procedure=cfg.splitting_procedure,
        filtering=cfg.dataset_filtering,
        feature_func_stpp=feature_funcs.get_features_func(cfg.saccade_predictors_funcs),
        feature_func_dur=feature_funcs.get_features_func(cfg.duration_predictors_funcs),
        division_factor_space=cfg.division_factor_space,
        division_factor_time=cfg.division_factor_time,
        division_factor_durations=cfg.division_factor_durations,
        past_timesteps_duration_baseline_k=cfg.past_timesteps_duration_baseline_k,
        cfg=cfg,
        datadir=datapath
    )

    train_ds = MecoDatasetW(mode="train", **dataset_kwargs)
    val_ds = MecoDatasetW(mode="valid", **dataset_kwargs)

    if cfg.subset:
        train_ds = Subset(train_ds, range(cfg.subset_size))
        val_ds = Subset(val_ds, range(int(cfg.subset_size * 0.2)))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.nworkers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.nworkers,
        pin_memory=True,
    )

    model = MarkedPointProcess(
        duration_prediction_func=cfg.duration_predictors_funcs,
        hawkes_predictors_func=cfg.saccade_predictors_funcs,
        model_type=cfg.model_type,
        cfg=cfg,
        logger=logger,
    ).to(device)    

    conv_param_names = {"gamma_alpha", "gamma_beta"}
    conv_params = []
    other_params = []

    for name, param in model.named_parameters():
        if name in conv_param_names:
            conv_params.append(param)
        else:
            other_params.append(param)

    # 2) Build optimizer depending on cfg.optimizer
    if cfg.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            [
                {"params": other_params, "lr": cfg.learning_rate},
                {"params": conv_params, "lr": 10 * cfg.learning_rate},
            ],
            weight_decay=cfg.weight_decay,
        )
    else:  # SGDNesterov
        optimizer = torch.optim.SGD(
            [
                {"params": other_params, "lr": cfg.learning_rate},
                {"params": conv_params, "lr": 10 * cfg.learning_rate},
            ],
            weight_decay=cfg.weight_decay,
            momentum=0.9,
            nesterov=True,
        )

    NegativeLogLikelihood = (
        set_saccadesNLL(cfg=cfg)
        if cfg.model_type == "saccade"
        else DurationNLL(distribution=cfg.dur_likelihood)
    )

    trainer = TrainerW(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        criterion=NegativeLogLikelihood,
        run_dir=datapath,
        logging=logger,
        device=device,
        patience=cfg.patience,
    )

    trainer.train(train_loader, val_loader=val_loader, epochs=cfg.epochs)