import torch
import numpy as np
import json

from spatio_temporal_reading.loss.loss_dur import NegLogLikelihood, NegLogLikelihood_np, NegLogLikelihoodCov_np

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

        var_dur = variances["var_dur"]

        self.std_dur = torch.tensor([np.sqrt(var_dur)], device=self.device, dtype=torch.float32)

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
                input_model, durations_target = item
    
                weights, durations_model, std = self.model(input_model)

                loss = NegLogLikelihoodCov_np(
                        weights=weights,
                        durations_model=durations_model,
                        durations_target=durations_target,
                        std=std,
                        std_dur=self.std_dur
                    )

                losses.append(loss)

        losses_np = np.concatenate(losses)
        return losses_np