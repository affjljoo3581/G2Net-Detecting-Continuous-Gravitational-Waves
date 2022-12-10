from __future__ import annotations

import argparse
import glob
import os

import h5py
import numpy as np
import tqdm

from utils import create_input_image_from_SFTs


def main(args: argparse.Namespace):
    os.makedirs(args.output_directory, exist_ok=True)

    for filename in tqdm.tqdm(glob.glob(os.path.join(args.directory, "*.hdf5"))):
        with h5py.File(filename) as data:
            name = list(data.keys())[0]
            L1_SFTs = np.array(data[f"{name}/L1/SFTs"], dtype=np.complex128)
            H1_SFTs = np.array(data[f"{name}/H1/SFTs"], dtype=np.complex128)

            image = create_input_image_from_SFTs(
                L1_SFTs, H1_SFTs, args.max_length, args.stride, args.scale
            )
            np.save(os.path.join(args.output_directory, f"{name}.npy"), image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--max-length", default=4096, type=int)
    parser.add_argument("--stride", default=32, type=int)
    parser.add_argument("--scale", default=1e22, type=float)
    parser.add_argument("--output-directory", default="resources/test")
    main(parser.parse_args())
