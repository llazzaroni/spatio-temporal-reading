from pathlib import Path
import torch
from torch.utils.data import DataLoader
from captum.attr import IntegratedGradients
import json
import random
import numpy as np

from spatio_temporal_reading.data.data import MecoDataset
from spatio_temporal_reading.model.model import TransformerCov
from spatio_temporal_reading.model.modelLM import TransformerCovLM
from attribution.utils.compute_means import baseline_tensor_sacc
from attribution.utils.statistics import one_sample_posx, one_sample_posy, one_sample_sacc, one_sample_posx_raw, one_sample_posy_raw, one_sample_sacc_raw


def compute_attr(args):
    datadir = Path(args.data)

    # First, create the dataset
    if args.filtering == "filtered":
        if not args.augment:
            train_ds = MecoDataset(mode="test", filtering="filtered", datadir=datadir)
        else:
            raise NotImplementedError
    else:
        if not args.augment:
            train_ds = MecoDataset(mode="test", filtering="raw", datadir=datadir)

    # Then, load the model
    if args.model_type == "saccades":
        device = "cpu"
        if not args.augment:
            if args.filtering == "filtered":
                checkpoint_path = datadir / "saccade" / "TRANSFORMER_FILTERED_COV" / "best_model.pt"
            else:
                checkpoint_path = datadir / "saccade" / "TRANSFORMER_RAW_COV" / "best_model.pt"
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            config = checkpoint["config"]
            if isinstance(config, dict):
                config = dict(config)
            elif hasattr(config, "__dict__"):
                config = vars(config).copy()
            else:
                config = dict(config)
            config.setdefault("dropout", 0.0)
            model = TransformerCov(**config)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model.eval()
        else:
            checkpoint_path = datadir / "saccade" / "TRANSFORMER_FILTERED_COV_LM" / "best_model.pt"
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            config = checkpoint["config"]
            if isinstance(config, dict):
                config = dict(config)
            elif hasattr(config, "__dict__"):
                config = vars(config).copy()
            else:
                config = dict(config)
            config.setdefault("dropout", 0.0)
            model = TransformerCovLM(**config)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model.eval()
    else:
        raise NotImplementedError
    

    dataloader_kwargs = dict(
        batch_size=1,
        num_workers=1,
        pin_memory=False,
        persistent_workers=False,
    )

    train_loader = DataLoader(
        train_ds,
        shuffle=False,
        **dataloader_kwargs,
    )

    if args.filtering == "filtered":
        sacc = np.zeros([8])
        posx = np.zeros([8])
        posy = np.zeros([8])
    else:
        sacc = np.zeros([9])
        posx = np.zeros([9])
        posy = np.zeros([9])


    for i, item in enumerate(train_loader):
        input_model, positions_target, saccades_target = item

        if args.model_type == "saccades":
            if not args.augment:
                    input_baseline = baseline_tensor_sacc(datadir, input_model.shape[1] - 1, input_model, args)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        
        indices = random.sample(range(1, input_model.shape[1]), k=10)

        if args.filtering == "filtered":
            for idx in indices:
                sacc += np.array(one_sample_sacc(input_model, input_baseline, idx, model))
                posx += np.array(one_sample_posx(input_model, input_baseline, idx, model))
                posy += np.array(one_sample_posy(input_model, input_baseline, idx, model))
        else:
            for idx in indices:
                sacc += np.array(one_sample_sacc_raw(input_model, input_baseline, idx, model))
                posx += np.array(one_sample_posx_raw(input_model, input_baseline, idx, model))
                posy += np.array(one_sample_posy_raw(input_model, input_baseline, idx, model))

        print(i)

    print(sacc / ((i + 1) * 10) * 100)
    print(posx / ((i + 1) * 10) * 100)
    print(posy / ((i + 1) * 10) * 100)


    return model, train_ds
