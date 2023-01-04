from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def main(args: argparse.Namespace):
    # Load the predictions and get their true labels. If `label-in-name` is true, then
    # extract the label from their name.
    preds = [pd.read_csv(x, index_col="id") for x in args.predictions]
    if args.label_in_name:
        labels = preds[0].copy()
        labels.target = labels.index.map(lambda x: int(int(x[:-4].split("-")[-1]) > 0))
    else:
        labels = pd.read_csv(args.labels, index_col="id")

    # Remove abnormal labels and rearrange with label indices.
    labels = labels[labels.target != -1]
    preds = [pred.loc[labels.index] for pred in preds]

    # Calculate the scores and sort by the scores.
    scores = [roc_auc_score(labels.target, pred.target) for pred in preds]
    highest_indices = np.argsort(scores)[::-1]

    weights, ensembled = np.ones(1), preds[highest_indices[0]].target
    score_history = []

    for i in highest_indices[1:]:
        # Get all combinations with ensembled prediction.
        combinations = []
        for j in np.linspace(-0.5, 0.51, 100):
            combined = (1 - j) * ensembled + j * preds[i].target
            combinations.append((j, combined, roc_auc_score(labels.target, combined)))

        # Find the best weight and update the ensembling.
        best_weight, ensembled, score = max(combinations, key=lambda x: x[2])
        weights = np.append(weights * (1 - best_weight), best_weight)
        score_history.append(score)

    # Match the weights with their name and print the formatted JSON string.
    weights = {
        os.path.basename(args.predictions[i]): weight
        for i, weight in zip(highest_indices, weights)
    }
    print(json.dumps(weights))

    if args.plot_history:
        plt.plot(score_history)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", nargs="+")
    parser.add_argument("--labels", default="resources/competition/train_labels.csv")
    parser.add_argument("--label-in-name", action="store_true", default=False)
    parser.add_argument("--plot-history", action="store_true", default=False)
    main(parser.parse_args())
