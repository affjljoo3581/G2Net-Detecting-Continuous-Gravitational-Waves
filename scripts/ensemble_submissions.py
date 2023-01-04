from __future__ import annotations

import argparse
import json
import os

import pandas as pd


def main(args: argparse.Namespace):
    with open(args.weights) as fp:
        weights = json.load(fp)
    preds = [pd.read_csv(x, index_col="id") for x in args.predictions]

    ensembled = sum(
        weights[os.path.basename(filename)] * pred
        for filename, pred in zip(args.predictions, preds)
    )
    ensembled.target = ensembled.target - ensembled.target.min()
    ensembled.target = ensembled.target / ensembled.target.max()
    ensembled.to_csv(args.weights.replace(".json", ".csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", nargs="+")
    parser.add_argument("--weights", default="ensemble-weights-train.json")
    main(parser.parse_args())
