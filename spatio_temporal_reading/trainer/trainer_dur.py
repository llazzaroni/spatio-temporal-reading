import torch
import numpy as np
import json
import time

from spatio_temporal_reading.loss.loss_dur import NegLogLikelihood, NegLogLikelihoodCov

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
            device="cpu",
            use_amp=False,
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
        self.use_amp = bool(use_amp and device == "cuda")
        self.non_blocking = device == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        with open(datapath / "variances.json", "r") as f:
            variances = json.load(f)

        var_dur = variances["var_dur"]

        self.std_dur = torch.tensor([np.sqrt(var_dur)], device=self.device, dtype=torch.float32)

    def _sync_cuda(self):
        if self.device == "cuda":
            torch.cuda.synchronize()

    def train(self):

        best_val_loss = float("inf")
        train_start = time.perf_counter()

        for epoch in range(self.epochs):
            epoch_start = time.perf_counter()
            losses_epoch = []
            fwd_times = []
            bwd_times = []
            for i, item in enumerate(self.train_loader):
                self.optimizer.zero_grad(set_to_none=True)

                # Unpack the item
                input_model, durations_target = item
                input_model = input_model.to(self.device, non_blocking=self.non_blocking)
                durations_target = durations_target.to(self.device, non_blocking=self.non_blocking)
                
                self._sync_cuda()
                fwd_start = time.perf_counter()
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    weights, durations_model = self.model(input_model)

                    loss = NegLogLikelihood(
                        weights=weights,
                        durations_model=durations_model,
                        durations_target=durations_target,
                        std_dur=self.std_dur
                    )
                self._sync_cuda()
                fwd_times.append(time.perf_counter() - fwd_start)

                losses_epoch.append(loss.item() / durations_model.shape[1])

                self._sync_cuda()
                bwd_start = time.perf_counter()
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                self._sync_cuda()
                bwd_times.append(time.perf_counter() - bwd_start)

            print(f"reached epoch {epoch}")

            avg_epoch_loss = np.array(losses_epoch).mean()
            print(f"avg loss on training set loss: {avg_epoch_loss}")
            epoch_wall = time.perf_counter() - epoch_start
            total_fwd = float(np.sum(fwd_times))
            total_bwd = float(np.sum(bwd_times))
            denom = epoch_wall if epoch_wall > 0 else 1e-12
            print(
                f"timing train epoch {epoch}: "
                f"iters={len(fwd_times)}, "
                f"forward_total={total_fwd:.2f}s ({100.0 * total_fwd / denom:.1f}%), "
                f"backward_total={total_bwd:.2f}s ({100.0 * total_bwd / denom:.1f}%), "
                f"epoch_total={epoch_wall:.2f}s"
            )

            val_loss = self.eval()
            print(f"eval loss: {val_loss}")

            if val_loss <= best_val_loss:
                best_val_loss = val_loss

                checkpoint = {
                    "model_state_dict": self.model.state_dict(),
                    "config": self.config,
                }

                torch.save(checkpoint, self.outputpath / "best_model.pt")
        print(f"total training wall time: {time.perf_counter() - train_start:.2f}s")


    def eval(self):
        losses = []

        with torch.no_grad():
            for i, item in enumerate(self.val_loader):
                input_model, durations_target = item
                input_model = input_model.to(self.device, non_blocking=self.non_blocking)
                durations_target = durations_target.to(self.device, non_blocking=self.non_blocking)

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    weights, durations_model = self.model(input_model)

                    loss = NegLogLikelihood(
                        weights=weights,
                        durations_model=durations_model,
                        durations_target=durations_target,
                        std_dur=self.std_dur
                    )

                losses.append(loss.item() / durations_model.shape[1])

        return np.array(losses).mean()
    

class TrainerCov(Trainer):
    def train(self):
        best_val_loss = float("inf")
        train_start = time.perf_counter()

        for epoch in range(self.epochs):
            epoch_start = time.perf_counter()
            losses_epoch = []
            fwd_times = []
            bwd_times = []
            for i, item in enumerate(self.train_loader):
                self.optimizer.zero_grad(set_to_none=True)

                # Unpack the item
                input_model, durations_target = item
                input_model = input_model.to(self.device, non_blocking=self.non_blocking)
                durations_target = durations_target.to(self.device, non_blocking=self.non_blocking)

                self._sync_cuda()
                fwd_start = time.perf_counter()
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    weights, durations_model, std = self.model(input_model)

                    loss = NegLogLikelihoodCov(
                        weights=weights,
                        durations_model=durations_model,
                        durations_target=durations_target,
                        std=std,
                        std_dur=self.std_dur
                    )
                self._sync_cuda()
                fwd_times.append(time.perf_counter() - fwd_start)

                losses_epoch.append(loss.item() / durations_model.shape[1])

                self._sync_cuda()
                bwd_start = time.perf_counter()
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                self._sync_cuda()
                bwd_times.append(time.perf_counter() - bwd_start)

            print(f"reached epoch {epoch}")

            avg_epoch_loss = np.array(losses_epoch).mean()
            print(f"avg loss on training set loss: {avg_epoch_loss}")
            epoch_wall = time.perf_counter() - epoch_start
            total_fwd = float(np.sum(fwd_times))
            total_bwd = float(np.sum(bwd_times))
            denom = epoch_wall if epoch_wall > 0 else 1e-12
            print(
                f"timing train epoch {epoch}: "
                f"iters={len(fwd_times)}, "
                f"forward_total={total_fwd:.2f}s ({100.0 * total_fwd / denom:.1f}%), "
                f"backward_total={total_bwd:.2f}s ({100.0 * total_bwd / denom:.1f}%), "
                f"epoch_total={epoch_wall:.2f}s"
            )

            val_loss = self.eval()
            print(f"eval loss: {val_loss}")

            if val_loss <= best_val_loss:
                best_val_loss = val_loss

                checkpoint = {
                    "model_state_dict": self.model.state_dict(),
                    "config": self.config,
                }

                torch.save(checkpoint, self.outputpath / "best_model.pt")
        print(f"total training wall time: {time.perf_counter() - train_start:.2f}s")


    def eval(self):
        losses = []

        with torch.no_grad():
            for i, item in enumerate(self.val_loader):
                input_model, durations_target = item
                input_model = input_model.to(self.device, non_blocking=self.non_blocking)
                durations_target = durations_target.to(self.device, non_blocking=self.non_blocking)

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    weights, durations_model, std = self.model(input_model)

                    loss = NegLogLikelihoodCov(
                        weights=weights,
                        durations_model=durations_model,
                        durations_target=durations_target,
                        std=std,
                        std_dur=self.std_dur
                    )

                losses.append(loss.item() / durations_model.shape[1])

        return np.array(losses).mean()
