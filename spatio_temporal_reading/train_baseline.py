from torch.utils.data import DataLoader, Subset
import torch
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Any, Dict
import json

from submodule.src.dataset import feature_funcs
from spatio_temporal_reading.submodule_wrappers.dataset import MecoDatasetW
from submodule.src.dataset.utils import collate_fn
from submodule.src.model.neural import MarkedPointProcess
from submodule.src.model.saccades.log_likelihood import set_saccadesNLL
from submodule.src.model.durations.log_likelihood import DurationNLL
from spatio_temporal_reading.submodule_wrappers.trainer import TrainerW

class DummyLogger:
    def info(self, *args, **kwargs):
        pass

def select_best_checkpoint(root):
    best = None
    best_loss = float("inf")
    for run in root.iterdir():
        metrics = run / "metrics.json"
        weights = run / "best_model_baseline.pt"
        if metrics.exists() and weights.exists():
            with metrics.open() as f:
                val_loss = json.load(f).get("val_loss")
            if val_loss is not None and val_loss < best_loss:
                best_loss = val_loss
                best = run
    print("Found the best model at", best)
    return best


def main(datapath, cfg, load_model, load_path):

    device = "cpu"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
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

    print("Model device:", next(model.parameters()).device)

    # Load the model   
    if load_model:
        root = datapath / load_path
        load_dir = select_best_checkpoint(root)
        checkpoint = torch.load(
            f=load_dir / "best_model_baseline.pt",
            map_location=device,
        )
        model.load_state_dict(state_dict=checkpoint, strict=cfg.strict_load)

        model.to(device)

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

    result_dir = datapath / cfg.directory_name
    result_dir.mkdir(parents=True, exist_ok=True)
    dir_string = str(cfg.model_type) + "_" + str(cfg.batch_size) + "_" + str(cfg.learning_rate) + "_" + str(cfg.weight_decay)
    if cfg.model_type == "duration":
        dir_string += "_" + str(cfg.alpha_g) + "_" + str(cfg.beta_g)
    experiment_dir = result_dir / dir_string

    trainer = TrainerW(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        criterion=NegativeLogLikelihood,
        run_dir=experiment_dir,
        logging=logger,
        device=device,
        patience=cfg.patience,
    )

    trainer.train(train_loader, val_loader=val_loader, epochs=cfg.epochs)

    cfg_path = experiment_dir / "config.json"
    with cfg_path.open("w") as f:
        json.dump(asdict(cfg), f, indent=2, default=str)