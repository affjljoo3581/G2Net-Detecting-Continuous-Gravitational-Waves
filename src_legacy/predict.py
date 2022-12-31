from __future__ import annotations

import argparse
import glob
import os

import pandas as pd
import torch
import tqdm
from torch.utils.data import DataLoader

from data import G2NetTestDataset


def augment_images(images: torch.Tensor, use_tta: False) -> list[torch.Tensor]:
    if not use_tta:
        return [images]
    return [images, images.flip(2), images.flip(3), images.flip(2, 3)]


@torch.inference_mode()
def main(args: argparse.Namespace):
    model = torch.jit.load(args.model).cuda().eval()

    filenames = sorted(glob.glob(os.path.join(args.directory, "*.npy")))
    dataloader = DataLoader(G2NetTestDataset(filenames), batch_size=args.batch_size)

    outputs = []
    for batch in tqdm.tqdm(dataloader):
        probs = [model(x) for x in augment_images(batch["images"].cuda(), args.use_tta)]
        outputs += torch.stack(probs).squeeze(2).sigmoid().mean(0).tolist()

    preds = pd.DataFrame({"id": filenames, "target": outputs})
    preds["id"] = preds["id"].map(lambda x: os.path.basename(x)[:-4])

    filename = args.model.replace(".pt", "-tta.csv" if args.use_tta else ".csv")
    preds.to_csv(filename, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--directory", default="resources/competition/test")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--use-tta", action="store_true", default=False)
    main(parser.parse_args())
