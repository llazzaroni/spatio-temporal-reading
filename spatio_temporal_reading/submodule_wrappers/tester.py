import torch
import numpy as np

class Tester:
    def __init__(
            self,
            model,
            test_loader,
            criterion,
            cfg
    ):
        self.model = model
        self.test_loader = test_loader
        self.device = "cpu"
        self.cfg = cfg
        self.criterion = criterion

    def test(self):

        self.model.eval()
        losses = []

        with torch.no_grad():
            for (
                input_features_stpp,
                input_features_dur,
                history_points,
                current_point,
                current_dur,
                boxes,
                fixations_cond,
            ) in self.test_loader:

                loss, loss_detached, _ = self.forward_pass(
                    model=self.model,
                    input_features_stpp=input_features_stpp,
                    input_features_dur=input_features_dur,
                    history_points=history_points,
                    current_point=current_point,
                    current_dur=current_dur,
                )
            
                losses.append(loss_detached.numpy())

        losses_np = np.concatenate(losses)
        return losses_np
    
    def forward_pass(
        self,
        model,
        input_features_stpp,
        input_features_dur,
        history_points,
        current_point,
        current_dur,
        radius=None,
        reduce="mean",
    ):
        
        input_features_stpp = input_features_stpp.to(self.device)
        input_features_dur = input_features_dur.to(self.device)
        current_dur = current_dur.to(self.device)
        current_point = current_point.to(self.device)
        history_points = history_points.to(self.device)
        current_dur = current_dur.to(self.device)

        neural_pars = model(input_features_stpp, input_features_dur, current_dur)

        mu, alpha, beta, sigma, means, dur_rate = neural_pars

        kwargs_outputs = {
            "baseline_kwargs": {"mu": mu},
            "temporal_kwargs": {"alpha": alpha, "beta": beta},
            "spatial_kwargs": {"mean": means, "sigma": sigma},
        }

        if self.cfg.model_type == "saccade":

            loss, probability = self.criterion(
                current_point,
                history_points,
                **kwargs_outputs,
                compute_probability=False,
                radius=radius,
                reduce="none",
            )

        elif self.cfg.model_type == "duration":
            loss = self.criterion(
                current_dur[:, 0].unsqueeze(-1), dur_rate, reduce=reduce
            )

        if alpha is not None and sigma is not None:
            self.curr_alpha_avg = alpha.mean().item()
            self.curr_sigma = sigma.item()

        if self.cfg.model_type == "saccade":
            return loss, loss.detach(), probability
        elif self.cfg.model_type == "duration":
            return loss, loss.detach(), None