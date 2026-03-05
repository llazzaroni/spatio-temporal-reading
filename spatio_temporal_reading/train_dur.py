from torch.utils.data import DataLoader
import torch
import pandas as pd
import matplotlib.pyplot as plt

from spatio_temporal_reading.data.data_dur import MecoDataset_dur
from spatio_temporal_reading.data.dataLM_dur import MecoDatasetLM_dur
from spatio_temporal_reading.model.model import SimpleModel, TransformerCov
from spatio_temporal_reading.model.modelLM import SimpleModelLM, TransformerCovLM, TransformerCovLM_conv
from spatio_temporal_reading.loss.loss_dur import NegLogLikelihood
from spatio_temporal_reading.trainer.trainer_dur import Trainer, TrainerCov

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def main(datapath, outputpath, args):
    device = get_device()

    if not args.augment:
        train_ds = MecoDataset_dur(mode="train", filtering=args.filtering, datadir=datapath)
        val_ds = MecoDataset_dur(mode="valid", filtering=args.filtering, datadir=datapath)
    else:
        train_ds = MecoDatasetLM_dur(mode="train", filtering=args.filtering, datadir=datapath)
        val_ds = MecoDatasetLM_dur(mode="valid", filtering=args.filtering, datadir=datapath)

    model_config = {
        "model_type": args.model_type,
        "d_in": train_ds.d_in_saccade,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_admixture_components": args.n_components,
        "max_len": train_ds.max_len,
        "H": args.heads,
        "dropout": float(args.dropout)
    }

    dataloader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        **dataloader_kwargs,
    )

    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **dataloader_kwargs,
    )

    if not args.augment:
        if args.cov:
            model = TransformerCov(
                model_type=model_config["model_type"],
                d_in=train_ds.d_in_saccade,
                n_layers=model_config["n_layers"],
                d_model=model_config["d_model"],
                n_admixture_components=model_config["n_admixture_components"],
                max_len=model_config["max_len"],
                H=model_config["H"],
                dropout=model_config["dropout"]
            ).to(device)
        else:
            model = SimpleModel(
                model_type=model_config["model_type"],
                d_in=train_ds.d_in_saccade,
                n_layers=model_config["n_layers"],
                d_model=model_config["d_model"],
                n_admixture_components=model_config["n_admixture_components"],
                max_len=model_config["max_len"],
                H=model_config["H"],
                dropout=model_config["dropout"]
            ).to(device)
    else:
        if args.cov:
            if args.conv:
                model = TransformerCovLM_conv(
                    model_type=model_config["model_type"],
                    d_in=train_ds.d_in_saccade,
                    n_layers=model_config["n_layers"],
                    d_model=model_config["d_model"],
                    n_admixture_components=model_config["n_admixture_components"],
                    max_len=model_config["max_len"],
                    H=model_config["H"],
                    dropout=model_config["dropout"]
                ).to(device)
            else:
                model = TransformerCovLM(
                    model_type=model_config["model_type"],
                    d_in=train_ds.d_in_saccade,
                    n_layers=model_config["n_layers"],
                    d_model=model_config["d_model"],
                    n_admixture_components=model_config["n_admixture_components"],
                    max_len=model_config["max_len"],
                    H=model_config["H"],
                    dropout=model_config["dropout"]
                ).to(device)
        else:
            model = SimpleModelLM(
                model_type=model_config["model_type"],
                d_in=train_ds.d_in_saccade,
                n_layers=model_config["n_layers"],
                d_model=model_config["d_model"],
                n_admixture_components=model_config["n_admixture_components"],
                max_len=model_config["max_len"],
                H=model_config["H"],
                dropout=model_config["dropout"]
            ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if args.cov:
        trainer = TrainerCov(
            train_loader,
            val_loader,
            args.epochs,
            model,
            optimizer,
            datapath,
            outputpath,
            model_config,
            device=device,
            use_amp=args.amp,
        )
    else:
        trainer = Trainer(
            train_loader,
            val_loader,
            args.epochs,
            model,
            optimizer,
            datapath,
            outputpath,
            model_config,
            device=device,
            use_amp=args.amp,
        )

    trainer.train()
