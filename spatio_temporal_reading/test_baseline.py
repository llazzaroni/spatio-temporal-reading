from torch.utils.data import DataLoader, Subset
import torch
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Any, Dict
import json
import numpy as np

from submodule.src.dataset import feature_funcs
from spatio_temporal_reading.submodule_wrappers.dataset import MecoDatasetW
from spatio_temporal_reading.submodule_wrappers.tester import Tester
from submodule.src.dataset.utils import collate_fn
from submodule.src.model.neural import MarkedPointProcess
from submodule.src.model.saccades.log_likelihood import set_saccadesNLL
from submodule.src.model.durations.log_likelihood import DurationNLL
from spatio_temporal_reading.launchers.baseline.grid_search_baseline import RunConfig


FOLDERS = ["RME_CSS_WS_FILTERED", "RME_CSS_WS_RAW", "BASE_LF_RAW", "BASE_SHP_RAW", "CSS_RAW", "poisson_raw_baseline", "RME_CSS_CHAR_LEN_FREQ_RAW", "RME_CSS_CHAR_WORD_LEN_FREQ_RAW", "RME_CSS_CS_RAW", "RME_CSS_DUR_RAW", "RME_CSS_FREQ_RAW", "RME_CSS_LEN_FREQ_RAW", "RME_CSS_LEN_RAW", "RME_CSS_RAW", "RME_CSS_WORD_LEN_FREQ_RAW", "RME_CSS_WS_RAW", "BASE_LF_FILTERED", "BASE_SHP_FILTERED", "CSS_FILTERED", "poisson_filtered_baseline", "RME_CSS_CHAR_LEN_FREQ_FILTERED", "RME_CSS_CHAR_WORD_LEN_FREQ_FILTERED", "RME_CSS_CS_FILTERED", "RME_CSS_DUR_FILTERED", "RME_CSS_FREQ_FILTERED", "RME_CSS_LEN_FREQ_FILTERED", "RME_CSS_LEN_FILTERED", "RME_CSS_FILTERED", "RME_CSS_WORD_LEN_FREQ_FILTERED", "RME_CSS_WS_FILTERED"]


class DummyLogger:
    def info(self, *args, **kwargs):
        pass

def test_model(datapath, cfg, checkpoint_dir):
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

    test_ds = MecoDatasetW(mode="test", **dataset_kwargs)

    test_loader = DataLoader(
        test_ds,
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

    checkpoint = torch.load(
        f=checkpoint_dir / "best_model_baseline.pt",
        map_location=device,
    )
    model.load_state_dict(state_dict=checkpoint, strict=cfg.strict_load)

    model.to(device)

    NegativeLogLikelihood = (
        set_saccadesNLL(cfg=cfg)
        if cfg.model_type == "saccade"
        else DurationNLL(distribution=cfg.dur_likelihood)
    )

    tester = Tester(
        model=model,
        test_loader=test_loader,
        criterion=NegativeLogLikelihood,
        cfg=cfg
    )

    losses = tester.test()
    return losses


def select_best_model(root):
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


def main(datapath, args):
    root = Path(args.models_path)

    for model_name in FOLDERS:
        model_path = root / model_name
        checkpoint_path = select_best_model(model_path)
        cfg_path = checkpoint_path / "config.json"
        with cfg_path.open() as f:
            cfg_json = json.load(f)
        cfg = RunConfig(**cfg_json)
        losses = test_model(datapath, cfg, checkpoint_path)

        outputpath = model_path / "negloglikelihoods.npy"
        np.save(outputpath, losses)
