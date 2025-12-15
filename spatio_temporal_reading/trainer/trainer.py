import torch
import numpy as np
import json

from spatio_temporal_reading.loss.loss import NegLogLikelihood

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
            config
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = EPOCHS
        self.model = model
        self.optimizer = optimizer
        self.datapath = datapath
        self.outputpath = outputpath
        self.config = config
        with open(datapath / "variances.json", "r") as f:
            variances = json.load(f)

        var_x = variances["var_x"]
        var_y = variances["var_y"]
        var_sacc = variances["var_sacc"]

        self.var_pos = torch.tensor([[var_x, 0], [0, var_y]])
        self.std_sacc = torch.tensor([np.sqrt(var_sacc)])

    def train(self):

        best_val_loss = float("inf")

        for epoch in range(self.epochs):
            losses_epoch = []
            for i, item in enumerate(self.train_loader):

                self.optimizer.zero_grad()

                # Unpack the item
                positions, durations, starting_times, saccades, reader_emb, features = item

                # Take the input to the model; exclude the last fixation in the scanpath
                input_model = torch.cat([
                    positions[:, 1:-1, :],
                    durations[:, 1:-1].unsqueeze(-1),
                    starting_times[:, 1:-1].unsqueeze(-1),
                    reader_emb[:, 1:-1, :],
                    features[:, 1:-1, :]],
                    dim=-1
                )
                
                # Pass the input through the model
                weights, positions_model, saccades_model = self.model(input_model)

                # Find the targets; exclude the first fixation in the scanpath
                positions_target = positions[:, 2:, :]
                saccades_target = saccades[:, 2:]

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
                positions, durations, starting_times, saccades, reader_emb, features = item

                input_model = torch.cat([
                    positions[:, 1:-1, :],
                    durations[:, 1:-1].unsqueeze(-1),
                    starting_times[:, 1:-1].unsqueeze(-1),
                    reader_emb[:, 1:-1, :],
                    features[:, 1:-1, :]],
                    dim=-1
                )

                weights, positions_model, saccades_model = self.model(input_model)

                positions_target = positions[:, 2:, :]
                saccades_target = saccades[:, 2:]

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

            