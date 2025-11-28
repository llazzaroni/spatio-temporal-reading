from torch.utils.data import DataLoader
import torch
import json
import numpy as np
import pandas as pd

from spatio_temporal_reading.data.data import MecoDataset
from spatio_temporal_reading.model.model import SimpleModel
from spatio_temporal_reading.loss.loss import NegLogLikelihood

def main(datapath):

    meco_df = pd.read_csv(datapath / "hp_augmented_meco_100_1000_1_10_model.csv").copy()
    print(pd.isna(meco_df["x"]).to_numpy().sum())
    print(pd.isna(meco_df["y"]).to_numpy().sum())
    print(pd.isna(meco_df["freq"]).to_numpy().sum())
    print(pd.isna(meco_df["dur"]).to_numpy().sum())
    print(pd.isna(meco_df["saccade"]).to_numpy().sum())
    print(pd.isna(meco_df["char_level_surp"]).to_numpy().sum())
    print(pd.isna(meco_df["word_level_surprisal"]).to_numpy().sum())
    print(pd.isna(meco_df["len"]).to_numpy().sum())

    '''

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
        d_in=train_ds.d_in,
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

    for epoch in range(30):
        for i, input_item in enumerate(train_loader):

            optimizer.zero_grad()

            # input_item has dimensions (n_batch, length_sequence, in_dim)

            print(torch.isnan(input_item).sum())

            # Take up to the last item as input of the model
            input_model = input_item[:, :-1, :]

            # Pass the input through the model
            weights, positions_model, saccades_model = model(input_model)

            # Take from the second item as targets; predict displacements and saccades
            positions_target = input_item[:, 1:, :2] - input_item[:, :-1, :2]
            saccades_target = input_item[:, 1:, 3]

            loss = NegLogLikelihood(
                weights=weights,
                positions_model=positions_model,
                saccades_model=saccades_model,
                positions=positions_target,
                saccades=saccades_target,
                cov_pos=var_pos,
                std_sacc=std_sacc
            )

            #print(loss)
            print(f"reached step {i}")

            loss.backward()

            optimizer.step()

    '''
