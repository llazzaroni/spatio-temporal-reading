from torch.utils.data import DataLoader
import torch

from spatio_temporal_reading.data.data_baseline import MecoDataset_baseline
from spatio_temporal_reading.data import feature_func
from spatio_temporal_reading.model.model_baseline import MarkedPointProcess
from spatio_temporal_reading.loss.loss_baseline.log_likelihood_saccade import set_saccadesNLL
from spatio_temporal_reading.loss.loss_baseline.log_likelihood_duration import DurationNLL
from spatio_temporal_reading.trainer.trainer_baseline import Trainer

# Parameters:
# Optimizer (nestorov vs adam)
# Filtering (raw vs filtered)
# Feature functions

def main(datapath, outputpath, args):

    train_ds = MecoDataset_baseline(
        mode="train",
        filtering="raw",
        feature_func_stpp=feature_func.past_position_features_reader_char_word_len_freq,
        feature_func_dur=feature_func.dur_model_reader_char_word_len_freq_features,
        datadir=datapath
    )
    val_ds = MecoDataset_baseline(
        mode="valid",
        filtering="raw",
        feature_func_stpp=feature_func.past_position_features_reader_char_word_len_freq,
        feature_func_dur=feature_func.dur_model_reader_char_word_len_freq_features,
        datadir=datapath
    )

    # config = {
    #     "model_type": args.model_type,
    #     "d_in": train_ds.d_in_saccade,
    #     "d_model": args.d_model,
    #     "n_layers": args.n_layers,
    #     "n_admixture_components": args.n_components,
    #     "max_len": train_ds.max_len
    # }

    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    model = MarkedPointProcess(
        duration_prediction_func="dur_model_reader_char_word_len_freq_features",
        hawkes_predictors_func="past_position_features_reader_char_word_len_freq",
        model_type=args.model_type,
        dataset_filtering="raw"
    )

    conv_param_names = {"gamma_alpha", "gamma_beta"}
    conv_params = []
    other_params = []

    for name, param in model.named_parameters():
        if name in conv_param_names:
            conv_params.append(param)
        else:
            other_params.append(param)

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            [
                {"params": other_params, "lr": 1e-3},
                {"params": conv_params, "lr": 10 * 1e-3},
            ],
            weight_decay=0.0,
        )
    else:  # SGDNesterov
        optimizer = torch.optim.SGD(
            [
                {"params": other_params, "lr": 1e-3},
                {"params": conv_params, "lr": 10 * 1e-3},
            ],
            weight_decay=0.0,
            momentum=0.9,
            nesterov=True,
        )

    NegativeLogLikelihood = (
        set_saccadesNLL(saccade_likelihood="ExtendedHawkesProcess")
        if args.model_type == "saccade"
        else DurationNLL(distribution="normal")
    )

    trainer = Trainer(
        model_type=args.model_type,
        model=model,
        optimizer=optimizer,
        criterion=NegativeLogLikelihood,
        run_dir=run_dir,
        logging=logger,
        device="cpu",
        patience=cfg.patience,
    )

    trainer.train(train_loader, val_loader=val_loader, epochs=cfg.epochs) 
