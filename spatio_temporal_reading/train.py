from torch.utils.data import DataLoader
import torch
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from spatio_temporal_reading.data.data import MecoDataset
from spatio_temporal_reading.model.model import SimpleModel
from spatio_temporal_reading.loss.loss import NegLogLikelihood

def main(datapath):

    meco_df = pd.read_csv(datapath / "hp_augmented_meco_100_1000_1_10_model.csv").copy()

    train_ds = MecoDataset(mode="train", datadir=datapath)
    val_ds = MecoDataset(mode="valid", datadir=datapath)

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
        model_type="saccade",
        d_in=train_ds.d_in_saccade,
        n_layers=2,
        d_model=30,
        n_admixture_components=5
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    with open(datapath / "variances.json", "r") as f:
        variances = json.load(f)

    var_x = variances["var_x"]
    var_y = variances["var_y"]
    var_sacc = variances["var_sacc"]

    var_pos = torch.tensor([[var_x, 0], [0, var_y]])
    std_sacc = torch.tensor([np.sqrt(var_sacc)])

    losses = []

    for epoch in range(30):
        losses_epoch = 0
        for i, item in enumerate(train_loader):

            optimizer.zero_grad()

            # Unpack the item
            positions, durations, starting_times, saccades, reader_emb, features = item

            # Take the input to the model; exclude the last fixation in the scanpath
            input_model = torch.cat([
                positions[:, :-1, :],
                durations[:, :-1].unsqueeze(-1),
                starting_times[:, :-1].unsqueeze(-1),
                reader_emb[:, :-1, :],
                features[:, :-1, :]],
                dim=-1
            )
            
            # Pass the input through the model
            weights, positions_model, saccades_model = model(input_model)

            # Find the targets; exclude the first fixation in the scanpath
            positions_target = positions[:, 1:, :]
            saccades_target = saccades[:, 1:]

            loss = NegLogLikelihood(
                weights=weights,
                positions_model=positions_model,
                saccades_model=saccades_model,
                positions=positions_target,
                saccades=saccades_target,
                cov_pos=var_pos,
                std_sacc=std_sacc
            )

            losses_epoch += loss

            losses.append(loss)

            loss.backward()

            optimizer.step()

        print(f"reached epoch {epoch}!! zio pera")
        print(f"loss: {losses_epoch}")

    plt.plot(losses)
        
