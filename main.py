import argparse
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", message="'pin_memory' argument is set as true but not supported on MPS")

# Make the submodule importable when running from the repo root.
ROOT = Path(__file__).resolve().parent
SUBMODULE = ROOT / "submodule"
SUBMODULE_SRC = SUBMODULE / "src"
for path in (SUBMODULE, SUBMODULE_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data_preprocessing import data
from exploration import exploration
from spatio_temporal_reading import train, train_baseline, visualize_sequence

def main(args):
    if args.data:
        datapath = Path(args.data)
    if args.output:
        outputpath = Path(args.output)
    if args.checkpoint_path:
        checkpoint_path = Path(args.checkpoint_path)
    if args.include_indices:
        data.include_index(datapath)
    if args.make_plots:
        exploration.make_plots(datapath, outputpath)
    if args.train:
        train.main(
            datapath=datapath,
            outputpath=outputpath,
            args=args
        )
    if args.train_baseline:
        train_baseline.main(
            datapath=datapath,
            outputpath=outputpath,
            args=args
        )
    if args.visualize_model:
        visualize_sequence.main(datapath, checkpoint_path, outputpath, args)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--include-indices", default=False, action="store_true")
    p.add_argument("--augment-fixation", default=False, action="store_true")
    p.add_argument("--make-plots", default=False, action="store_true")
    p.add_argument("--train", default=False, action="store_true")
    p.add_argument("--train-baseline", default=False, action="store_true")
    p.add_argument("--visualize-model", default=False, action="store_true")
    p.add_argument("--train_index", type=int, default=0)
    p.add_argument("--val_index", type=int, default=0)
    p.add_argument("--model-type", type=str, default="saccade")
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--d-model", type=int, default=30)
    p.add_argument("--n-components", type=int, default=20)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--data")
    p.add_argument("--output")
    p.add_argument("--checkpoint-path")
    args = p.parse_args()
    main(args)
