from torch.utils.data import DataLoader
import torch
import numpy as np
from pathlib import Path

from spatio_temporal_reading.data.data import MecoDataset
from spatio_temporal_reading.data.dataLM import MecoDatasetLM
from spatio_temporal_reading.model.model import SimpleModel, TransformerCov
from spatio_temporal_reading.model.modelLM import SimpleModelLM, TransformerCovLM
from spatio_temporal_reading.trainer.tester import Tester, TesterCov

def get_device():
    return "cpu"


def load_model(checkpoint_path, args, device):
    # Explicitly allow full checkpoint loading (needed for older saves)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    if args.augment:
        if args.cov:
            model = TransformerCovLM(**config)
        else:
            model = SimpleModelLM(**config)
    else:
        if args.cov:
            model = TransformerCov(**config)
        else:
            model = SimpleModel(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model

def main(datapath, args):

    if not args.augment:
        test_ds = MecoDataset(mode="test", filtering=args.filtering, datadir=datapath)
    else:
        test_ds = MecoDatasetLM(mode="test", filtering=args.filtering, datadir=datapath)

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
        tester = Tester(
            test_loader=test_loader,
            model=model,
            datapath=datapath,
            device=device
        )
    losses = tester.test()

    checkpoint_path = Path(checkpoint_path)
    outputpath = checkpoint_path.parent / "negloglikelihoods.npy"
    np.save(outputpath, losses)
