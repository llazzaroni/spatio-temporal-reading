import argparse
from pathlib import Path

from data_preprocessing import data
from exploration import exploration
from spatio_temporal_reading import train

def main(args):
    if args.data:
        datapath = Path(args.data)
    if args.output:
        outputpath = Path(args.output)
    if args.include_indices:
        data.include_index(datapath)
    if args.make_plots:
        exploration.make_plots(datapath, outputpath)
    if args.train:
        train.main(datapath)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build cycling winners dataset.")
    p.add_argument("--include-indices", default=False)
    p.add_argument("--augment-fixation", default=False)
    p.add_argument("--make-plots", default=False)
    p.add_argument("--train", default=False)
    p.add_argument("--data")
    p.add_argument("--output")
    args = p.parse_args()
    main(args)