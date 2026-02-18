import torch
import numpy as np
import json

from spatio_temporal_reading.loss.loss import NegLogLikelihood, NegLogLikelihood_np, NegLogLikelihoodCov_np

class Tester:

    def __init__(
            self,
            test_loader,
            model,
            datapath,
            device="cpu"
    ):
        self.test_loader = test_loader
        self.model = model
        self.device = device

        with open(datapath / "variances.json", "r") as f:
            variances = json.load(f)

        var_x = variances["var_x"]
        var_y = variances["var_y"]
        var_sacc = variances["var_sacc"]

        self.var_pos = torch.tensor([[var_x, 0], [0, var_y]], device=self.device)
        self.std_sacc = torch.tensor([np.sqrt(var_sacc)], device=self.device)


    def test(self):
        losses = []

        with torch.no_grad():
            for i, item in enumerate(self.test_loader):
                input_model, positions_target, saccades_target = item
                input_model = input_model.to(self.device)
                positions_target = positions_target.to(self.device)
                saccades_target = saccades_target.to(self.device)

                weights, positions_model, saccades_model = self.model(input_model)

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
            
class TesterCov(Tester):
    def test(self):
        losses = []

        with torch.no_grad():
            for i, item in enumerate(self.test_loader):
                input_model, positions_target, saccades_target = item
                input_model = input_model.to(self.device)
                positions_target = positions_target.to(self.device)
                saccades_target = saccades_target.to(self.device)

                weights, positions_model, saccades_model, covariances2D, covariancesSacc = self.model(input_model)

                loss = NegLogLikelihoodCov_np(
                    covariances2D=covariances2D,
                    covariancesSacc=covariancesSacc,
                    weights=weights,
                    positions_model=positions_model,
                    saccades_model=saccades_model,
                    positions=positions_target,
                    saccades=saccades_target
                )

                losses.append(loss)

        losses_np = np.concatenate(losses)
        return losses_np