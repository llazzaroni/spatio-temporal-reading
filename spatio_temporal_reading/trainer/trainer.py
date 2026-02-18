import torch
import numpy as np
import json

from spatio_temporal_reading.loss.loss import NegLogLikelihood, NegLogLikelihoodCov

class Trainer:

    def __init__(
            self,
            train_loader,
            val_loader,
            EPOCHS,
            model,
            optimizer,
            datapath,
            outputpath,
            config,
            device="cpu"
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = EPOCHS
        self.model = model
        self.optimizer = optimizer
        self.datapath = datapath
        self.outputpath = outputpath
        self.config = config
        self.device = device
        with open(datapath / "variances.json", "r") as f:
            variances = json.load(f)

        var_x = variances["var_x"]
        var_y = variances["var_y"]
        var_sacc = variances["var_sacc"]

        self.var_pos = torch.tensor([[var_x, 0], [0, var_y]], device=self.device, dtype=torch.float32)
        self.std_sacc = torch.tensor([np.sqrt(var_sacc)], device=self.device, dtype=torch.float32)

    def train(self):

        best_val_loss = float("inf")

        for epoch in range(self.epochs):
            losses_epoch = []
            for i, item in enumerate(self.train_loader):

                self.optimizer.zero_grad()

                # Unpack the item
                input_model, positions_target, saccades_target = item
                input_model = input_model.to(self.device)
                positions_target = positions_target.to(self.device)
                saccades_target = saccades_target.to(self.device)

                weights, positions_model, saccades_model = self.model(input_model)

                loss = NegLogLikelihood(
                    weights=weights,
                    positions_model=positions_model,
                    saccades_model=saccades_model,
                    positions=positions_target,
                    saccades=saccades_target,
                    cov_pos=self.var_pos,
                    std_sacc=self.std_sacc
                )

                losses_epoch.append(loss.item() / positions_model.shape[1])

                loss.backward()

                self.optimizer.step()

            print(f"reached epoch {epoch}")

            avg_epoch_loss = np.array(losses_epoch).mean()
            print(f"avg loss on training set loss: {avg_epoch_loss}")

            val_loss = self.eval()
            print(f"eval loss: {val_loss}")

            if val_loss <= best_val_loss:
                best_val_loss = val_loss

                checkpoint = {
                    "model_state_dict": self.model.state_dict(),
                    "config": self.config,
                }

                torch.save(checkpoint, self.outputpath / "best_model.pt")


    def eval(self):
        losses = []

        with torch.no_grad():
            for i, item in enumerate(self.val_loader):
                input_model, positions_target, saccades_target = item
                input_model = input_model.to(self.device)
                positions_target = positions_target.to(self.device)
                saccades_target = saccades_target.to(self.device)

                weights, positions_model, saccades_model = self.model(input_model)

                loss = NegLogLikelihood(
                    weights=weights,
                    positions_model=positions_model,
                    saccades_model=saccades_model,
                    positions=positions_target,
                    saccades=saccades_target,
                    cov_pos=self.var_pos,
                    std_sacc=self.std_sacc
                )

                losses.append(loss.item() / positions_model.shape[1])

        return np.array(losses).mean()
    

class TrainerCov(Trainer):
    def train(self):
        best_val_loss = float("inf")

        for epoch in range(self.epochs):
            losses_epoch = []
            for i, item in enumerate(self.train_loader):

                self.optimizer.zero_grad()

                # Unpack the item
                input_model, positions_target, saccades_target = item
                input_model = input_model.to(self.device)
                positions_target = positions_target.to(self.device)
                saccades_target = saccades_target.to(self.device)

                weights, positions_model, saccades_model, covariances2D, covariancesSacc = self.model(input_model)

                loss = NegLogLikelihoodCov(
                    covariances2D=covariances2D,
                    covariancesSacc=covariancesSacc,
                    weights=weights,
                    positions_model=positions_model,
                    saccades_model=saccades_model,
                    positions=positions_target,
                    saccades=saccades_target
                )

                losses_epoch.append(loss.item() / positions_model.shape[1])

                loss.backward()

                self.optimizer.step()

            print(f"reached epoch {epoch}")

            avg_epoch_loss = np.array(losses_epoch).mean()
            print(f"avg loss on training set loss: {avg_epoch_loss}")

            val_loss = self.eval()
            print(f"eval loss: {val_loss}")

            if val_loss <= best_val_loss:
                best_val_loss = val_loss

                checkpoint = {
                    "model_state_dict": self.model.state_dict(),
                    "config": self.config,
                }

                torch.save(checkpoint, self.outputpath / "best_model.pt")


    def eval(self):
        losses = []

        with torch.no_grad():
            for i, item in enumerate(self.val_loader):
                input_model, positions_target, saccades_target = item
                input_model = input_model.to(self.device)
                positions_target = positions_target.to(self.device)
                saccades_target = saccades_target.to(self.device)

                weights, positions_model, saccades_model, covariances2D, covariancesSacc = self.model(input_model)

                loss = NegLogLikelihoodCov(
                    covariances2D=covariances2D,
                    covariancesSacc=covariancesSacc,
                    weights=weights,
                    positions_model=positions_model,
                    saccades_model=saccades_model,
                    positions=positions_target,
                    saccades=saccades_target
                )

                losses.append(loss.item() / positions_model.shape[1])

        return np.array(losses).mean()

            
