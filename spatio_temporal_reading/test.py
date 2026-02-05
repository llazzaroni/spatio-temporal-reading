from torch.utils.data import DataLoader
import torch
import numpy as np
from pathlib import Path

from spatio_temporal_reading.data.data import MecoDataset
from spatio_temporal_reading.model.model import SimpleModel
from spatio_temporal_reading.trainer.tester import Tester


def load_model(checkpoint_path):
    # Explicitly allow full checkpoint loading (needed for older saves)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = SimpleModel(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def main(datapath, args):
    
    test_ds = MecoDataset(mode="test", filtering=args.filtering, datadir=datapath)

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )

    checkpoint_path = args.checkpoint_path
    model = load_model(checkpoint_path)

    tester = Tester(
        test_loader=test_loader,
        model=model,
        datapath=datapath
    )
    losses = tester.test()

    checkpoint_path = Path(checkpoint_path)
    outputpath = checkpoint_path.parent / "negloglikelihoods.npy"
    np.save(outputpath, losses)