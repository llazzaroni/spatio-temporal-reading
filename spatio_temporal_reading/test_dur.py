from torch.utils.data import DataLoader
import torch
import numpy as np
from pathlib import Path

from spatio_temporal_reading.data.data_dur import MecoDataset_dur
from spatio_temporal_reading.data.dataLM_dur import MecoDatasetLM_dur
from spatio_temporal_reading.model.model import SimpleModel, TransformerCov
from spatio_temporal_reading.model.modelLM import SimpleModelLM, TransformerCovLM, TransformerCovLM_conv
from spatio_temporal_reading.trainer.tester_dur import Tester, TesterCov

def get_device():
    return "cpu"


def load_model(checkpoint_path, args, device):
    # Explicitly allow full checkpoint loading (needed for older saves)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    if isinstance(config, dict):
        config = dict(config)
    elif hasattr(config, "__dict__"):
        config = vars(config).copy()
    else:
        config = dict(config)
    config.setdefault("dropout", 0.0)
    if args.augment:
        if args.cov:
            if args.conv:
                raise NotImplementedError
            else:
               model = TransformerCovLM(**config)
        else:
            raise NotImplementedError
    else:
        if args.cov:
            model = TransformerCov(**config)
        else:
            raise NotImplementedError
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model

def main(datapath, args):

    if not args.augment:
        test_ds = MecoDataset_dur(mode="test", filtering=args.filtering, datadir=datapath)
    else:
        if args.conv:
            raise NotImplementedError
        else:
            test_ds = MecoDatasetLM_dur(mode="test", filtering=args.filtering, datadir=datapath)

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    device = get_device()
    checkpoint_path = args.checkpoint_path
    model = load_model(checkpoint_path, args, device)

    if args.cov:
        tester = TesterCov(
            test_loader=test_loader,
            model=model,
            datapath=datapath,
            device=device
        )
    else:
        raise NotImplementedError
    losses = tester.test()

    checkpoint_path = Path(checkpoint_path)
    outputpath = checkpoint_path.parent / "negloglikelihoods.npy"
    np.save(outputpath, losses)
