import argparse
import sys
from pathlib import Path

# Make the submodule importable when running from the repo root.
ROOT = Path(__file__).resolve().parent
SUBMODULE = ROOT / "submodule"
SUBMODULE_SRC = SUBMODULE / "src"
SUBMODULE_SCRIPTS = SUBMODULE / "scripts"
for path in (SUBMODULE, SUBMODULE_SRC, SUBMODULE_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

def main(args):
    if args.data:
        datapath = Path(args.data)
    if args.output:
        outputpath = Path(args.output)
    if args.checkpoint_path:
        checkpoint_path = Path(args.checkpoint_path)
    if args.include_indices:
        from data_preprocessing import data
        data.include_index(datapath)
    if args.make_plots:
        from exploration import exploration
        exploration.make_plots(datapath, outputpath)
    if args.train:
        if args.model_type == "durations":
            from spatio_temporal_reading import train_dur
            train_dur.main(
                datapath=datapath,
                outputpath=outputpath,
                args=args
            )
        else:
            from spatio_temporal_reading import train_sacc
            train_sacc.main(
                datapath=datapath,
                outputpath=outputpath,
                args=args
            )
    if args.train_baseline:
        from spatio_temporal_reading.launchers.baseline import grid_search_baseline
        grid_search_baseline.main(datapath)        
    if args.visualize_model:
        from spatio_temporal_reading import visualize_sequence
        visualize_sequence.main(datapath, checkpoint_path, outputpath, args)
    if args.test:
        if args.model_type == "saccades":
            from spatio_temporal_reading import test_sacc
            test_sacc.main(
                datapath=datapath,
                args=args
            )
        else:
            from spatio_temporal_reading import test_dur
            test_dur.main(
                datapath=datapath,
                args=args
            )
    if args.test_baseline:
        from spatio_temporal_reading import test_baseline
        test_baseline.main(
            datapath=datapath,
            args=args
        )
    if args.include_tokens:
        from data_preprocessing import data
        data.include_tokens(datapath)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--include-indices", default=False, action="store_true")
    p.add_argument("--include-tokens", default=False, action="store_true")
    p.add_argument("--augment-fixation", default=False, action="store_true")
    p.add_argument("--make-plots", default=False, action="store_true")
    p.add_argument("--train", default=False, action="store_true")
    p.add_argument("--train-baseline", default=False, action="store_true")
    p.add_argument("--visualize-model", default=False, action="store_true")
    p.add_argument("--train_index", type=int, default=0)
    p.add_argument("--val_index", type=int, default=0)
    p.add_argument("--model-type", type=str, default="saccades")
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--d-model", type=int, default=30)
    p.add_argument("--heads", type=int, default=5)
    p.add_argument("--n-components", type=int, default=20)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--amp", default=False, action="store_true")
    p.add_argument("--data")
    p.add_argument("--output")
    p.add_argument("--checkpoint-path")
    p.add_argument("--filtering", default="filtered")
    p.add_argument("--test", default=False, action="store_true")
    p.add_argument("--test-baseline", default=False, action="store_true")
    p.add_argument("--models-path")
    p.add_argument("--augment", default=False, action="store_true")
    p.add_argument("--cov", default=False, action="store_true")
    p.add_argument("--conv", default=False, action="store_true")
    p.add_argument("--dropout", default=0.0)
    args = p.parse_args()
    main(args)