from __future__ import annotations

import argparse
import glob
import os

import pandas as pd
import torch
import tqdm
from torch.utils.data import DataLoader

from data import G2NetTestDataset


def augment_input_images(
    images: torch.Tensor,
    use_flip_tta: bool = False,
    num_vertical_shuffle: int | None = None,
    num_horizontal_shuffle: int | None = None,
) -> list[torch.Tensor]:
    images_list = [images]
    if use_flip_tta:
        images_list.extend((images.flip(2), images.flip(3), images.flip(2, 3)))

    for images in list(images_list):
        if num_vertical_shuffle is not None:
            for _ in range(num_vertical_shuffle):
                i = torch.arange(images.size(3)) + torch.normal(0, 2, (images.size(3),))
                images_list.append(images[:, :, :, i.argsort()].contiguous())
        if num_horizontal_shuffle is not None:
            for _ in range(num_horizontal_shuffle):
                i = torch.arange(images.size(2)) + torch.normal(0, 3, (images.size(2),))
                images_list.append(images[:, :, i.argsort(), :].contiguous())

    return images_list


@torch.inference_mode()
def main(args: argparse.Namespace):
    model = torch.jit.load(args.model).cuda().eval()

    filenames = sorted(glob.glob(os.path.join(args.directory, "*.npy")))
    dataloader = DataLoader(G2NetTestDataset(filenames), batch_size=args.batch_size)

    outputs = []
    for batch in tqdm.tqdm(dataloader):
        augmented_images = augment_input_images(
            batch["images"].cuda(),
            args.use_flip_tta,
            args.num_vertical_shuffle,
            args.num_horizontal_shuffle,
        )
        probs = [model(x) for x in augmented_images]
        outputs += torch.stack(probs).squeeze(2).sigmoid().mean(0).tolist()

    preds = pd.DataFrame({"id": filenames, "target": outputs})
    preds["id"] = preds["id"].map(lambda x: os.path.basename(x)[:-4])

    postfix_list = []
    if args.use_flip_tta:
        postfix_list.append("-flip4")
    if args.num_vertical_shuffle is not None:
        postfix_list.append(f"-vs{args.num_vertical_shuffle}")
    if args.num_horizontal_shuffle is not None:
        postfix_list.append(f"-hs{args.num_horizontal_shuffle}")

    filename = args.model.replace(".pt", "".join(postfix_list) + ".csv")
    preds.to_csv(filename, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--directory", default="resources/competition/test")
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--use-flip-tta", action="store_true", default=False)
    parser.add_argument("--num-vertical-shuffle", type=int)
    parser.add_argument("--num-horizontal-shuffle", type=int)
    main(parser.parse_args())
