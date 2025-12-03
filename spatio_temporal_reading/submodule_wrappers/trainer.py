import torch

from submodule.src.trainer import Trainer

class TrainerW(Trainer):
    def train(self, train_loader, val_loader, epochs):
        for epoch in range(1, epochs + 1):
            print("Reached epoch", epoch)
            if self.lr_rescaling < 1 and epoch != 1:
                self.optimizer.param_groups[0]["lr"] = (
                    self.optimizer.param_groups[0]["lr"] * self.lr_rescaling
                )

            stats_train = self.train_epoch(train_loader, epoch)
            self.loss_tracker_global["train"][self.cfg.model_type].append(stats_train)

            train_loss_mean = stats_train[0]
            print(f"train mean loss: {train_loss_mean}")

            stats_val = self.validate_epoch(val_loader)
            self.loss_tracker_global["val"][self.cfg.model_type].append(stats_val)

            val_loss_mean = stats_val[0]
            print(f"val mean loss: {val_loss_mean}")

            if self.check_early_stopping(stats_val[0], epoch=epoch):
                break

            if self.cfg.model_type == "saccade" and (
                self.curr_alpha_avg <= 0.01 or self.curr_sigma <= 1e-9
            ):
                self.model.initialize_parameters()
                self.model.to(self.device)
                learning_rate = self.optimizer.param_groups[0]["lr"]
                weight_decay = self.optimizer.param_groups[0]["weight_decay"]
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
                )
        