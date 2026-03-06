import argparse
import sys
from pathlib import Path

# Make package imports work when running from inside `attribution/`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attribution.utils.compute_means import compute_means_sacc_filtered
from attribution.eval_model.compute_attr import compute_attr

def main(args):
    if args.compute_means:
        if not args.augment:
            if args.model_type == "saccades":
                compute_means_sacc_filtered(args)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    if args.attributions:
        compute_attr(args)





if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--compute-means", default=False, action="store_true")
    p.add_argument("--attributions", default=False, action="store_true")
    p.add_argument("--data")
    p.add_argument("--model-type", default="saccades")
    p.add_argument("--augment", default=False, action="store_true")
    p.add_argument("--filtering", default="filtered")
    args = p.parse_args()
    main(args)
