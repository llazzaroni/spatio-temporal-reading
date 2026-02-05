from torch.utils.data import DataLoader
import torch
import pandas as pd
import matplotlib.pyplot as plt

from spatio_temporal_reading.data.data import MecoDataset
from spatio_temporal_reading.model.model import SimpleModel
from spatio_temporal_reading.loss.loss import NegLogLikelihood
from spatio_temporal_reading.trainer.trainer import Trainer

def main(datapath, outputpath, args):

    train_ds = MecoDataset(mode="train", filtering=args.filtering, datadir=datapath)
    val_ds = MecoDataset(mode="valid", filtering=args.filtering, datadir=datapath)

    model_config = {
        "model_type": args.model_type,
        "d_in": train_ds.d_in_saccade,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_admixture_components": args.n_components,
        "max_len": train_ds.max_len
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

    model = SimpleModel(
        model_type=model_config["model_type"],
        d_in=train_ds.d_in_saccade,
        n_layers=model_config["n_layers"],
        d_model=model_config["d_model"],
        n_admixture_components=model_config["n_admixture_components"],
        max_len=model_config["max_len"]
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainer = Trainer(train_loader, val_loader, args.epochs, model, optimizer, datapath, outputpath, model_config)

    trainer.train()

        
