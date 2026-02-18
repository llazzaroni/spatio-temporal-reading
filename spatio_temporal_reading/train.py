from torch.utils.data import DataLoader
import torch
import pandas as pd
import matplotlib.pyplot as plt

from spatio_temporal_reading.data.data import MecoDataset
from spatio_temporal_reading.data.dataLM import MecoDatasetLM
from spatio_temporal_reading.model.model import SimpleModel, TransformerCov
from spatio_temporal_reading.model.modelLM import SimpleModelLM, TransformerCovLM
from spatio_temporal_reading.loss.loss import NegLogLikelihood
from spatio_temporal_reading.trainer.trainer import Trainer, TrainerCov

def get_device():
    return "cpu"

def main(datapath, outputpath, args):

    if not args.augment:
        train_ds = MecoDataset(mode="train", filtering=args.filtering, datadir=datapath)
        val_ds = MecoDataset(mode="valid", filtering=args.filtering, datadir=datapath)
    else:
        train_ds = MecoDatasetLM(mode="train", filtering=args.filtering, datadir=datapath)
        val_ds = MecoDatasetLM(mode="valid", filtering=args.filtering, datadir=datapath)

    model_config = {
        "model_type": args.model_type,
        "d_in": train_ds.d_in_saccade,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_admixture_components": args.n_components,
        "max_len": train_ds.max_len,
        "H": args.heads
    }

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

    device = get_device()

    if not args.augment:
        if args.cov:
            model = TransformerCov(
                model_type=model_config["model_type"],
                d_in=train_ds.d_in_saccade,
                n_layers=model_config["n_layers"],
                d_model=model_config["d_model"],
                n_admixture_components=model_config["n_admixture_components"],
                max_len=model_config["max_len"],
                H=model_config["H"]
            ).to(device)
        else:
            model = SimpleModel(
                model_type=model_config["model_type"],
                d_in=train_ds.d_in_saccade,
                n_layers=model_config["n_layers"],
                d_model=model_config["d_model"],
                n_admixture_components=model_config["n_admixture_components"],
                max_len=model_config["max_len"],
                H=model_config["H"]
            ).to(device)
    else:
        if args.cov:
            model = TransformerCovLM(
                model_type=model_config["model_type"],
                d_in=train_ds.d_in_saccade,
                n_layers=model_config["n_layers"],
                d_model=model_config["d_model"],
                n_admixture_components=model_config["n_admixture_components"],
                max_len=model_config["max_len"],
                H=model_config["H"]
            ).to(device)
        else:
            model = SimpleModelLM(
                model_type=model_config["model_type"],
                d_in=train_ds.d_in_saccade,
                n_layers=model_config["n_layers"],
                d_model=model_config["d_model"],
                n_admixture_components=model_config["n_admixture_components"],
                max_len=model_config["max_len"],
                H=model_config["H"]
            ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if args.cov:
        trainer = TrainerCov(train_loader, val_loader, args.epochs, model, optimizer, datapath, outputpath, model_config, device=device)
    else:
        trainer = Trainer(train_loader, val_loader, args.epochs, model, optimizer, datapath, outputpath, model_config, device=device)

    trainer.train()
