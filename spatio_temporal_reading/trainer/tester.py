import torch
import numpy as np
import json

from spatio_temporal_reading.loss.loss import NegLogLikelihood
from spatio_temporal_reading.loss.loss import NegLogLikelihood_np

class Tester:

    def __init__(
            self,
            test_loader,
            model,
            datapath
    ):
        self.test_loader = test_loader
        self.model = model

        with open(datapath / "variances.json", "r") as f:
            variances = json.load(f)

        var_x = variances["var_x"]
        var_y = variances["var_y"]
        var_sacc = variances["var_sacc"]

        self.var_pos = torch.tensor([[var_x, 0], [0, var_y]])
        self.std_sacc = torch.tensor([np.sqrt(var_sacc)])


    def test(self):
        losses = []

        with torch.no_grad():
            for i, item in enumerate(self.test_loader):
                positions, durations, starting_times, saccades, reader_emb, features, BOS_token = item

                input_model = torch.cat([
                    positions[:, :-1, :],
                    durations[:, :-1].unsqueeze(-1),
                    starting_times[:, :-1].unsqueeze(-1),
                    reader_emb[:, :-1, :],
                    features[:, :-1, :],
                    BOS_token[:, :-1, :]],
                    dim=-1
                )

                weights, positions_model, saccades_model = self.model(input_model)

                positions_target = positions[:, 1:, :]
                saccades_target = saccades[:, 1:]

                loss = NegLogLikelihood_np(
                    weights=weights,
                    positions_model=positions_model,
                    saccades_model=saccades_model,
                    positions=positions_target,
                    saccades=saccades_target,
                    cov_pos=self.var_pos,
                    std_sacc=self.std_sacc
                )

                losses.append(loss)

        losses_np = np.concatenate(losses)
        return losses_np
            