# run_scarlett.py
from Scarlet.ltllearner import LTLlearner
import multiprocessing as mp
import sys
import argparse


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--csvname", type=str, default="scarlet_out.csv")
    parser.add_argument("--thres", type=int, default=0)
    return parser.parse_args()


def main():
    args = parser()
    args = vars(args)

    learner = LTLlearner(
        input_file=args["input_file"],   # your trace file
        timeout=args["timeout"],                     # optional
        csvname=args["csvname"],       # optional
        # optional, 0 = perfect separation
        thres=args["thres"]
    )
    learner.learn()


if __name__ == "__main__":
    # On macOS + Python 3.8+, spawn is default/safest. Make it explicit.
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Already set, fine.
        pass
    main()
