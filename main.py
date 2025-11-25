import argparse

from data_preprocessing import data
from exploration import exploration

def main(args):
    datapath = args.data
    if args.include_indices:
        data.include_index(datapath)
    if args.make_plots:
        outputpath = args.output
        exploration.make_plots(datapath, outputpath)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build cycling winners dataset.")
    p.add_argument("--include-indices", default=False)
    p.add_argument("--augment-fixation", default=False)
    p.add_argument("--make-plots", default=False)
    p.add_argument("--data")
    p.add_argument("--output")
    args = p.parse_args()
    main(args)